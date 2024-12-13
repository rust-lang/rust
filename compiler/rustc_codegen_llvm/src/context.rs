use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, c_char, c_uint};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::str;

use rustc_abi::{HasDataLayout, Size, TargetDataLayout, VariantIdx};
use rustc_codegen_ssa::back::versioned_llvm_target;
use rustc_codegen_ssa::base::{wants_msvc_seh, wants_wasm_eh};
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::errors as ssa_errors;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::base_n::{ALPHANUMERIC_ONLY, ToBaseN};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::PatchableFunctionEntry;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTypingEnv, LayoutError, LayoutOfHelpers,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::Session;
use rustc_session::config::{
    BranchProtection, CFGuard, CFProtection, CrateType, DebugInfo, FunctionReturn, PAuthKey, PacRet,
};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_target::spec::{HasTargetSpec, RelocModel, SmallDataThresholdSupport, Target, TlsModel};
use smallvec::SmallVec;

use crate::back::write::to_llvm_code_model;
use crate::callee::get_fn;
use crate::common::AsCCharPtr;
use crate::debuginfo::metadata::apply_vcall_visibility_metadata;
use crate::llvm::Metadata;
use crate::type_::Type;
use crate::value::Value;
use crate::{attributes, common, coverageinfo, debuginfo, llvm, llvm_util};

/// `TyCtxt` (and related cache datastructures) can't be move between threads.
/// However, there are various cx related functions which we want to be available to the builder and
/// other compiler pieces. Here we define a small subset which has enough information and can be
/// moved around more freely.
pub(crate) struct SCx<'ll> {
    pub llmod: &'ll llvm::Module,
    pub llcx: &'ll llvm::Context,
    pub isize_ty: &'ll Type,
}

impl<'ll> Borrow<SCx<'ll>> for FullCx<'ll, '_> {
    fn borrow(&self) -> &SCx<'ll> {
        &self.scx
    }
}

impl<'ll, 'tcx> Deref for FullCx<'ll, 'tcx> {
    type Target = SimpleCx<'ll>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.scx
    }
}

pub(crate) struct GenericCx<'ll, T: Borrow<SCx<'ll>>>(T, PhantomData<SCx<'ll>>);

impl<'ll, T: Borrow<SCx<'ll>>> Deref for GenericCx<'ll, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ll, T: Borrow<SCx<'ll>>> DerefMut for GenericCx<'ll, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub(crate) type SimpleCx<'ll> = GenericCx<'ll, SCx<'ll>>;

/// There is one `CodegenCx` per codegen unit. Each one has its own LLVM
/// `llvm::Context` so that several codegen units may be processed in parallel.
/// All other LLVM data structures in the `CodegenCx` are tied to that `llvm::Context`.
pub(crate) type CodegenCx<'ll, 'tcx> = GenericCx<'ll, FullCx<'ll, 'tcx>>;

pub(crate) struct FullCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub scx: SimpleCx<'ll>,
    pub use_dll_storage_attrs: bool,
    pub tls_model: llvm::ThreadLocalMode,

    pub codegen_unit: &'tcx CodegenUnit<'tcx>,

    /// Cache instances of monomorphic and polymorphic items
    pub instances: RefCell<FxHashMap<Instance<'tcx>, &'ll Value>>,
    /// Cache generated vtables
    pub vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), &'ll Value>>,
    /// Cache of constant strings,
    pub const_str_cache: RefCell<FxHashMap<String, &'ll Value>>,

    /// Cache of emitted const globals (value -> global)
    pub const_globals: RefCell<FxHashMap<&'ll Value, &'ll Value>>,

    /// List of globals for static variables which need to be passed to the
    /// LLVM function ReplaceAllUsesWith (RAUW) when codegen is complete.
    /// (We have to make sure we don't invalidate any Values referring
    /// to constants.)
    pub statics_to_rauw: RefCell<Vec<(&'ll Value, &'ll Value)>>,

    /// Statics that will be placed in the llvm.used variable
    /// See <https://llvm.org/docs/LangRef.html#the-llvm-used-global-variable> for details
    pub used_statics: Vec<&'ll Value>,

    /// Statics that will be placed in the llvm.compiler.used variable
    /// See <https://llvm.org/docs/LangRef.html#the-llvm-compiler-used-global-variable> for details
    pub compiler_used_statics: Vec<&'ll Value>,

    /// Mapping of non-scalar types to llvm types.
    pub type_lowering: RefCell<FxHashMap<(Ty<'tcx>, Option<VariantIdx>), &'ll Type>>,

    /// Mapping of scalar types to llvm types.
    pub scalar_lltypes: RefCell<FxHashMap<Ty<'tcx>, &'ll Type>>,

    /// Extra per-CGU codegen state needed when coverage instrumentation is enabled.
    pub coverage_cx: Option<coverageinfo::CguCoverageContext<'ll, 'tcx>>,
    pub dbg_cx: Option<debuginfo::CodegenUnitDebugContext<'ll, 'tcx>>,

    eh_personality: Cell<Option<&'ll Value>>,
    eh_catch_typeinfo: Cell<Option<&'ll Value>>,
    pub rust_try_fn: Cell<Option<(&'ll Type, &'ll Value)>>,

    intrinsics: RefCell<FxHashMap<&'static str, (&'ll Type, &'ll Value)>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,

    /// `codegen_static` will sometimes create a second global variable with a
    /// different type and clear the symbol name of the original global.
    /// `global_asm!` needs to be able to find this new global so that it can
    /// compute the correct mangled symbol name to insert into the asm.
    pub renamed_statics: RefCell<FxHashMap<DefId, &'ll Value>>,
}

fn to_llvm_tls_model(tls_model: TlsModel) -> llvm::ThreadLocalMode {
    match tls_model {
        TlsModel::GeneralDynamic => llvm::ThreadLocalMode::GeneralDynamic,
        TlsModel::LocalDynamic => llvm::ThreadLocalMode::LocalDynamic,
        TlsModel::InitialExec => llvm::ThreadLocalMode::InitialExec,
        TlsModel::LocalExec => llvm::ThreadLocalMode::LocalExec,
        TlsModel::Emulated => llvm::ThreadLocalMode::GeneralDynamic,
    }
}

pub(crate) unsafe fn create_module<'ll>(
    tcx: TyCtxt<'_>,
    llcx: &'ll llvm::Context,
    mod_name: &str,
) -> &'ll llvm::Module {
    let sess = tcx.sess;
    let mod_name = SmallCStr::new(mod_name);
    let llmod = unsafe { llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx) };

    let mut target_data_layout = sess.target.data_layout.to_string();
    let llvm_version = llvm_util::get_version();

    if llvm_version < (20, 0, 0) {
        if sess.target.arch == "aarch64" || sess.target.arch.starts_with("arm64") {
            // LLVM 20 defines three additional address spaces for alternate
            // pointer kinds used in Windows.
            // See https://github.com/llvm/llvm-project/pull/111879
            target_data_layout =
                target_data_layout.replace("-p270:32:32-p271:32:32-p272:64:64", "");
        }
        if sess.target.arch.starts_with("sparc") {
            // LLVM 20 updates the sparc layout to correctly align 128 bit integers to 128 bit.
            // See https://github.com/llvm/llvm-project/pull/106951
            target_data_layout = target_data_layout.replace("-i128:128", "");
        }
        if sess.target.arch.starts_with("mips64") {
            // LLVM 20 updates the mips64 layout to correctly align 128 bit integers to 128 bit.
            // See https://github.com/llvm/llvm-project/pull/112084
            target_data_layout = target_data_layout.replace("-i128:128", "");
        }
        if sess.target.arch.starts_with("powerpc64") {
            // LLVM 20 updates the powerpc64 layout to correctly align 128 bit integers to 128 bit.
            // See https://github.com/llvm/llvm-project/pull/118004
            target_data_layout = target_data_layout.replace("-i128:128", "");
        }
        if sess.target.arch.starts_with("wasm32") || sess.target.arch.starts_with("wasm64") {
            // LLVM 20 updates the wasm(32|64) layout to correctly align 128 bit integers to 128 bit.
            // See https://github.com/llvm/llvm-project/pull/119204
            target_data_layout = target_data_layout.replace("-i128:128", "");
        }
    }
    if llvm_version < (21, 0, 0) {
        if sess.target.arch == "nvptx64" {
            // LLVM 21 updated the default layout on nvptx: https://github.com/llvm/llvm-project/pull/124961
            target_data_layout = target_data_layout.replace("e-p6:32:32-i64", "e-i64");
        }
    }

    // Ensure the data-layout values hardcoded remain the defaults.
    {
        let tm = crate::back::write::create_informational_target_machine(tcx.sess, false);
        unsafe {
            llvm::LLVMRustSetDataLayoutFromTargetMachine(llmod, tm.raw());
        }

        let llvm_data_layout = unsafe { llvm::LLVMGetDataLayoutStr(llmod) };
        let llvm_data_layout =
            str::from_utf8(unsafe { CStr::from_ptr(llvm_data_layout) }.to_bytes())
                .expect("got a non-UTF8 data-layout from LLVM");

        if target_data_layout != llvm_data_layout {
            tcx.dcx().emit_err(crate::errors::MismatchedDataLayout {
                rustc_target: sess.opts.target_triple.to_string().as_str(),
                rustc_layout: target_data_layout.as_str(),
                llvm_target: sess.target.llvm_target.borrow(),
                llvm_layout: llvm_data_layout,
            });
        }
    }

    let data_layout = SmallCStr::new(&target_data_layout);
    unsafe {
        llvm::LLVMSetDataLayout(llmod, data_layout.as_ptr());
    }

    let llvm_target = SmallCStr::new(&versioned_llvm_target(sess));
    unsafe {
        llvm::LLVMRustSetNormalizedTarget(llmod, llvm_target.as_ptr());
    }

    let reloc_model = sess.relocation_model();
    if matches!(reloc_model, RelocModel::Pic | RelocModel::Pie) {
        unsafe {
            llvm::LLVMRustSetModulePICLevel(llmod);
        }
        // PIE is potentially more effective than PIC, but can only be used in executables.
        // If all our outputs are executables, then we can relax PIC to PIE.
        if reloc_model == RelocModel::Pie
            || tcx.crate_types().iter().all(|ty| *ty == CrateType::Executable)
        {
            unsafe {
                llvm::LLVMRustSetModulePIELevel(llmod);
            }
        }
    }

    // Linking object files with different code models is undefined behavior
    // because the compiler would have to generate additional code (to span
    // longer jumps) if a larger code model is used with a smaller one.
    //
    // See https://reviews.llvm.org/D52322 and https://reviews.llvm.org/D52323.
    unsafe {
        llvm::LLVMRustSetModuleCodeModel(llmod, to_llvm_code_model(sess.code_model()));
    }

    // If skipping the PLT is enabled, we need to add some module metadata
    // to ensure intrinsic calls don't use it.
    if !sess.needs_plt() {
        llvm::add_module_flag_u32(llmod, llvm::ModuleFlagMergeBehavior::Warning, "RtLibUseGOT", 1);
    }

    // Enable canonical jump tables if CFI is enabled. (See https://reviews.llvm.org/D65629.)
    if sess.is_sanitizer_cfi_canonical_jump_tables_enabled() && sess.is_sanitizer_cfi_enabled() {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Override,
            "CFI Canonical Jump Tables",
            1,
        );
    }

    // If we're normalizing integers with CFI, ensure LLVM generated functions do the same.
    // See https://github.com/llvm/llvm-project/pull/104826
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Override,
            "cfi-normalize-integers",
            1,
        );
    }

    // Enable LTO unit splitting if specified or if CFI is enabled. (See
    // https://reviews.llvm.org/D53891.)
    if sess.is_split_lto_unit_enabled() || sess.is_sanitizer_cfi_enabled() {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Override,
            "EnableSplitLTOUnit",
            1,
        );
    }

    // Add "kcfi" module flag if KCFI is enabled. (See https://reviews.llvm.org/D119296.)
    if sess.is_sanitizer_kcfi_enabled() {
        llvm::add_module_flag_u32(llmod, llvm::ModuleFlagMergeBehavior::Override, "kcfi", 1);

        // Add "kcfi-offset" module flag with -Z patchable-function-entry (See
        // https://reviews.llvm.org/D141172).
        let pfe =
            PatchableFunctionEntry::from_config(sess.opts.unstable_opts.patchable_function_entry);
        if pfe.prefix() > 0 {
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Override,
                "kcfi-offset",
                pfe.prefix().into(),
            );
        }

        // Add "kcfi-arity" module flag if KCFI arity indicator is enabled. (See
        // https://github.com/llvm/llvm-project/pull/117121.)
        if sess.is_sanitizer_kcfi_arity_enabled() {
            // KCFI arity indicator requires LLVM 21.0.0 or later.
            if llvm_version < (21, 0, 0) {
                tcx.dcx().emit_err(crate::errors::SanitizerKcfiArityRequiresLLVM2100);
            }

            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Override,
                "kcfi-arity",
                1,
            );
        }
    }

    // Control Flow Guard is currently only supported by MSVC and LLVM on Windows.
    if sess.target.is_like_msvc
        || (sess.target.options.os == "windows"
            && sess.target.options.env == "gnu"
            && sess.target.options.abi == "llvm")
    {
        match sess.opts.cg.control_flow_guard {
            CFGuard::Disabled => {}
            CFGuard::NoChecks => {
                // Set `cfguard=1` module flag to emit metadata only.
                llvm::add_module_flag_u32(
                    llmod,
                    llvm::ModuleFlagMergeBehavior::Warning,
                    "cfguard",
                    1,
                );
            }
            CFGuard::Checks => {
                // Set `cfguard=2` module flag to emit metadata and checks.
                llvm::add_module_flag_u32(
                    llmod,
                    llvm::ModuleFlagMergeBehavior::Warning,
                    "cfguard",
                    2,
                );
            }
        }
    }

    if let Some(BranchProtection { bti, pac_ret }) = sess.opts.unstable_opts.branch_protection {
        if sess.target.arch == "aarch64" {
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Min,
                "branch-target-enforcement",
                bti.into(),
            );
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Min,
                "sign-return-address",
                pac_ret.is_some().into(),
            );
            let pac_opts = pac_ret.unwrap_or(PacRet { leaf: false, pc: false, key: PAuthKey::A });
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Min,
                "branch-protection-pauth-lr",
                pac_opts.pc.into(),
            );
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Min,
                "sign-return-address-all",
                pac_opts.leaf.into(),
            );
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Min,
                "sign-return-address-with-bkey",
                u32::from(pac_opts.key == PAuthKey::B),
            );
        } else {
            bug!(
                "branch-protection used on non-AArch64 target; \
                  this should be checked in rustc_session."
            );
        }
    }

    // Pass on the control-flow protection flags to LLVM (equivalent to `-fcf-protection` in Clang).
    if let CFProtection::Branch | CFProtection::Full = sess.opts.unstable_opts.cf_protection {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Override,
            "cf-protection-branch",
            1,
        );
    }
    if let CFProtection::Return | CFProtection::Full = sess.opts.unstable_opts.cf_protection {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Override,
            "cf-protection-return",
            1,
        );
    }

    if sess.opts.unstable_opts.virtual_function_elimination {
        llvm::add_module_flag_u32(
            llmod,
            llvm::ModuleFlagMergeBehavior::Error,
            "Virtual Function Elim",
            1,
        );
    }

    // Set module flag to enable Windows EHCont Guard (/guard:ehcont).
    if sess.opts.unstable_opts.ehcont_guard {
        llvm::add_module_flag_u32(llmod, llvm::ModuleFlagMergeBehavior::Warning, "ehcontguard", 1);
    }

    match sess.opts.unstable_opts.function_return {
        FunctionReturn::Keep => {}
        FunctionReturn::ThunkExtern => {
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Override,
                "function_return_thunk_extern",
                1,
            );
        }
    }

    match (sess.opts.unstable_opts.small_data_threshold, sess.target.small_data_threshold_support())
    {
        // Set up the small-data optimization limit for architectures that use
        // an LLVM module flag to control this.
        (Some(threshold), SmallDataThresholdSupport::LlvmModuleFlag(flag)) => {
            llvm::add_module_flag_u32(
                llmod,
                llvm::ModuleFlagMergeBehavior::Error,
                &flag,
                threshold as u32,
            );
        }
        _ => (),
    };

    // Insert `llvm.ident` metadata.
    //
    // On the wasm targets it will get hooked up to the "producer" sections
    // `processed-by` information.
    #[allow(clippy::option_env_unwrap)]
    let rustc_producer =
        format!("rustc version {}", option_env!("CFG_VERSION").expect("CFG_VERSION"));
    let name_metadata = unsafe {
        llvm::LLVMMDStringInContext2(
            llcx,
            rustc_producer.as_c_char_ptr(),
            rustc_producer.as_bytes().len(),
        )
    };
    unsafe {
        llvm::LLVMAddNamedMetadataOperand(
            llmod,
            c"llvm.ident".as_ptr(),
            &llvm::LLVMMetadataAsValue(llcx, llvm::LLVMMDNodeInContext2(llcx, &name_metadata, 1)),
        );
    }

    // Emit RISC-V specific target-abi metadata
    // to workaround lld as the LTO plugin not
    // correctly setting target-abi for the LTO object
    // FIXME: https://github.com/llvm/llvm-project/issues/50591
    // If llvm_abiname is empty, emit nothing.
    let llvm_abiname = &sess.target.options.llvm_abiname;
    if matches!(sess.target.arch.as_ref(), "riscv32" | "riscv64") && !llvm_abiname.is_empty() {
        llvm::add_module_flag_str(
            llmod,
            llvm::ModuleFlagMergeBehavior::Error,
            "target-abi",
            llvm_abiname,
        );
    }

    // Add module flags specified via -Z llvm_module_flag
    for (key, value, merge_behavior) in &sess.opts.unstable_opts.llvm_module_flag {
        let merge_behavior = match merge_behavior.as_str() {
            "error" => llvm::ModuleFlagMergeBehavior::Error,
            "warning" => llvm::ModuleFlagMergeBehavior::Warning,
            "require" => llvm::ModuleFlagMergeBehavior::Require,
            "override" => llvm::ModuleFlagMergeBehavior::Override,
            "append" => llvm::ModuleFlagMergeBehavior::Append,
            "appendunique" => llvm::ModuleFlagMergeBehavior::AppendUnique,
            "max" => llvm::ModuleFlagMergeBehavior::Max,
            "min" => llvm::ModuleFlagMergeBehavior::Min,
            // We already checked this during option parsing
            _ => unreachable!(),
        };
        llvm::add_module_flag_u32(llmod, merge_behavior, key, *value);
    }

    llmod
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        codegen_unit: &'tcx CodegenUnit<'tcx>,
        llvm_module: &'ll crate::ModuleLlvm,
    ) -> Self {
        // An interesting part of Windows which MSVC forces our hand on (and
        // apparently MinGW didn't) is the usage of `dllimport` and `dllexport`
        // attributes in LLVM IR as well as native dependencies (in C these
        // correspond to `__declspec(dllimport)`).
        //
        // LD (BFD) in MinGW mode can often correctly guess `dllexport` but
        // relying on that can result in issues like #50176.
        // LLD won't support that and expects symbols with proper attributes.
        // Because of that we make MinGW target emit dllexport just like MSVC.
        // When it comes to dllimport we use it for constants but for functions
        // rely on the linker to do the right thing. Opposed to dllexport this
        // task is easy for them (both LD and LLD) and allows us to easily use
        // symbols from static libraries in shared libraries.
        //
        // Whenever a dynamic library is built on Windows it must have its public
        // interface specified by functions tagged with `dllexport` or otherwise
        // they're not available to be linked against. This poses a few problems
        // for the compiler, some of which are somewhat fundamental, but we use
        // the `use_dll_storage_attrs` variable below to attach the `dllexport`
        // attribute to all LLVM functions that are exported e.g., they're
        // already tagged with external linkage). This is suboptimal for a few
        // reasons:
        //
        // * If an object file will never be included in a dynamic library,
        //   there's no need to attach the dllexport attribute. Most object
        //   files in Rust are not destined to become part of a dll as binaries
        //   are statically linked by default.
        // * If the compiler is emitting both an rlib and a dylib, the same
        //   source object file is currently used but with MSVC this may be less
        //   feasible. The compiler may be able to get around this, but it may
        //   involve some invasive changes to deal with this.
        //
        // The flip side of this situation is that whenever you link to a dll and
        // you import a function from it, the import should be tagged with
        // `dllimport`. At this time, however, the compiler does not emit
        // `dllimport` for any declarations other than constants (where it is
        // required), which is again suboptimal for even more reasons!
        //
        // * Calling a function imported from another dll without using
        //   `dllimport` causes the linker/compiler to have extra overhead (one
        //   `jmp` instruction on x86) when calling the function.
        // * The same object file may be used in different circumstances, so a
        //   function may be imported from a dll if the object is linked into a
        //   dll, but it may be just linked against if linked into an rlib.
        // * The compiler has no knowledge about whether native functions should
        //   be tagged dllimport or not.
        //
        // For now the compiler takes the perf hit (I do not have any numbers to
        // this effect) by marking very little as `dllimport` and praying the
        // linker will take care of everything. Fixing this problem will likely
        // require adding a few attributes to Rust itself (feature gated at the
        // start) and then strongly recommending static linkage on Windows!
        let use_dll_storage_attrs = tcx.sess.target.is_like_windows;

        let tls_model = to_llvm_tls_model(tcx.sess.tls_model());

        let (llcx, llmod) = (&*llvm_module.llcx, llvm_module.llmod());

        let coverage_cx =
            tcx.sess.instrument_coverage().then(coverageinfo::CguCoverageContext::new);

        let dbg_cx = if tcx.sess.opts.debuginfo != DebugInfo::None {
            let dctx = debuginfo::CodegenUnitDebugContext::new(llmod);
            debuginfo::metadata::build_compile_unit_di_node(
                tcx,
                codegen_unit.name().as_str(),
                &dctx,
            );
            Some(dctx)
        } else {
            None
        };

        GenericCx(
            FullCx {
                tcx,
                scx: SimpleCx::new(llmod, llcx, tcx.data_layout.pointer_size),
                use_dll_storage_attrs,
                tls_model,
                codegen_unit,
                instances: Default::default(),
                vtables: Default::default(),
                const_str_cache: Default::default(),
                const_globals: Default::default(),
                statics_to_rauw: RefCell::new(Vec::new()),
                used_statics: Vec::new(),
                compiler_used_statics: Vec::new(),
                type_lowering: Default::default(),
                scalar_lltypes: Default::default(),
                coverage_cx,
                dbg_cx,
                eh_personality: Cell::new(None),
                eh_catch_typeinfo: Cell::new(None),
                rust_try_fn: Cell::new(None),
                intrinsics: Default::default(),
                local_gen_sym_counter: Cell::new(0),
                renamed_statics: Default::default(),
            },
            PhantomData,
        )
    }

    pub(crate) fn statics_to_rauw(&self) -> &RefCell<Vec<(&'ll Value, &'ll Value)>> {
        &self.statics_to_rauw
    }

    /// Extra state that is only available when coverage instrumentation is enabled.
    #[inline]
    #[track_caller]
    pub(crate) fn coverage_cx(&self) -> &coverageinfo::CguCoverageContext<'ll, 'tcx> {
        self.coverage_cx.as_ref().expect("only called when coverage instrumentation is enabled")
    }

    pub(crate) fn create_used_variable_impl(&self, name: &'static CStr, values: &[&'ll Value]) {
        let array = self.const_array(self.type_ptr(), values);

        let g = llvm::add_global(self.llmod, self.val_ty(array), name);
        llvm::set_initializer(g, array);
        llvm::set_linkage(g, llvm::Linkage::AppendingLinkage);
        llvm::set_section(g, c"llvm.metadata");
    }
}
impl<'ll> SimpleCx<'ll> {
    pub(crate) fn get_return_type(&self, ty: &'ll Type) -> &'ll Type {
        assert_eq!(self.type_kind(ty), TypeKind::Function);
        unsafe { llvm::LLVMGetReturnType(ty) }
    }
    pub(crate) fn get_type_of_global(&self, val: &'ll Value) -> &'ll Type {
        unsafe { llvm::LLVMGlobalGetValueType(val) }
    }
    pub(crate) fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        common::val_ty(v)
    }
}
impl<'ll> SimpleCx<'ll> {
    pub(crate) fn new(
        llmod: &'ll llvm::Module,
        llcx: &'ll llvm::Context,
        pointer_size: Size,
    ) -> Self {
        let isize_ty = llvm::Type::ix_llcx(llcx, pointer_size.bits());
        Self(SCx { llmod, llcx, isize_ty }, PhantomData)
    }
}

impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    pub(crate) fn get_metadata_value(&self, metadata: &'ll Metadata) -> &'ll Value {
        llvm::LLVMMetadataAsValue(self.llcx(), metadata)
    }

    // FIXME(autodiff): We should split `ConstCodegenMethods` to pull the reusable parts
    // onto a trait that is also implemented for GenericCx.
    pub(crate) fn get_const_i64(&self, n: u64) -> &'ll Value {
        let ty = unsafe { llvm::LLVMInt64TypeInContext(self.llcx()) };
        unsafe { llvm::LLVMConstInt(ty, n, llvm::False) }
    }

    pub(crate) fn get_function(&self, name: &str) -> Option<&'ll Value> {
        let name = SmallCStr::new(name);
        unsafe { llvm::LLVMGetNamedFunction((**self).borrow().llmod, name.as_ptr()) }
    }

    pub(crate) fn get_md_kind_id(&self, name: &str) -> llvm::MetadataKindId {
        unsafe {
            llvm::LLVMGetMDKindIDInContext(
                self.llcx(),
                name.as_ptr() as *const c_char,
                name.len() as c_uint,
            )
        }
    }

    pub(crate) fn create_metadata(&self, name: String) -> Option<&'ll Metadata> {
        Some(unsafe {
            llvm::LLVMMDStringInContext2(self.llcx(), name.as_ptr() as *const c_char, name.len())
        })
    }

    pub(crate) fn get_functions(&self) -> Vec<&'ll Value> {
        let mut functions = vec![];
        let mut func = unsafe { llvm::LLVMGetFirstFunction(self.llmod()) };
        while let Some(f) = func {
            functions.push(f);
            func = unsafe { llvm::LLVMGetNextFunction(f) }
        }
        functions
    }
}

impl<'ll, 'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), &'ll Value>> {
        &self.vtables
    }

    fn apply_vcall_visibility_metadata(
        &self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        vtable: &'ll Value,
    ) {
        apply_vcall_visibility_metadata(self, ty, poly_trait_ref, vtable);
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> &'ll Value {
        get_fn(self, instance)
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> &'ll Value {
        get_fn(self, instance)
    }

    fn eh_personality(&self) -> &'ll Value {
        // The exception handling personality function.
        //
        // If our compilation unit has the `eh_personality` lang item somewhere
        // within it, then we just need to codegen that. Otherwise, we're
        // building an rlib which will depend on some upstream implementation of
        // this function, so we just codegen a generic reference to it. We don't
        // specify any of the types for the function, we just make it a symbol
        // that LLVM can later use.
        //
        // Note that MSVC is a little special here in that we don't use the
        // `eh_personality` lang item at all. Currently LLVM has support for
        // both Dwarf and SEH unwind mechanisms for MSVC targets and uses the
        // *name of the personality function* to decide what kind of unwind side
        // tables/landing pads to emit. It looks like Dwarf is used by default,
        // injecting a dependency on the `_Unwind_Resume` symbol for resuming
        // an "exception", but for MSVC we want to force SEH. This means that we
        // can't actually have the personality function be our standard
        // `rust_eh_personality` function, but rather we wired it up to the
        // CRT's custom personality function, which forces LLVM to consider
        // landing pads as "landing pads for SEH".
        if let Some(llpersonality) = self.eh_personality.get() {
            return llpersonality;
        }

        let name = if wants_msvc_seh(self.sess()) {
            Some("__CxxFrameHandler3")
        } else if wants_wasm_eh(self.sess()) {
            // LLVM specifically tests for the name of the personality function
            // There is no need for this function to exist anywhere, it will
            // not be called. However, its name has to be "__gxx_wasm_personality_v0"
            // for native wasm exceptions.
            Some("__gxx_wasm_personality_v0")
        } else {
            None
        };

        let tcx = self.tcx;
        let llfn = match tcx.lang_items().eh_personality() {
            Some(def_id) if name.is_none() => self.get_fn_addr(ty::Instance::expect_resolve(
                tcx,
                self.typing_env(),
                def_id,
                ty::List::empty(),
                DUMMY_SP,
            )),
            _ => {
                let name = name.unwrap_or("rust_eh_personality");
                if let Some(llfn) = self.get_declared_value(name) {
                    llfn
                } else {
                    let fty = self.type_variadic_func(&[], self.type_i32());
                    let llfn = self.declare_cfn(name, llvm::UnnamedAddr::Global, fty);
                    let target_cpu = attributes::target_cpu_attr(self);
                    attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[target_cpu]);
                    llfn
                }
            }
        };
        self.eh_personality.set(Some(llfn));
        llfn
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, llfn: &'ll Value) {
        if let Some(attr) = attributes::frame_pointer_type_attr(self) {
            attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[attr]);
        }
    }

    fn apply_target_cpu_attr(&self, llfn: &'ll Value) {
        let mut attrs = SmallVec::<[_; 2]>::new();
        attrs.push(attributes::target_cpu_attr(self));
        attrs.extend(attributes::tune_cpu_attr(self));
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &attrs);
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        let entry_name = self.sess().target.entry_name.as_ref();
        if self.get_declared_value(entry_name).is_none() {
            Some(self.declare_entry_fn(
                entry_name,
                llvm::CallConv::from_conv(
                    self.sess().target.entry_abi,
                    self.sess().target.arch.borrow(),
                ),
                llvm::UnnamedAddr::Global,
                fn_type,
            ))
        } else {
            // If the symbol already exists, it is an error: for example, the user wrote
            // #[no_mangle] extern "C" fn main(..) {..}
            None
        }
    }
}

impl<'ll> CodegenCx<'ll, '_> {
    pub(crate) fn get_intrinsic(&self, key: &str) -> (&'ll Type, &'ll Value) {
        if let Some(v) = self.intrinsics.borrow().get(key).cloned() {
            return v;
        }

        self.declare_intrinsic(key).unwrap_or_else(|| bug!("unknown intrinsic '{}'", key))
    }

    fn insert_intrinsic(
        &self,
        name: &'static str,
        args: Option<&[&'ll llvm::Type]>,
        ret: &'ll llvm::Type,
    ) -> (&'ll llvm::Type, &'ll llvm::Value) {
        let fn_ty = if let Some(args) = args {
            self.type_func(args, ret)
        } else {
            self.type_variadic_func(&[], ret)
        };
        let f = self.declare_cfn(name, llvm::UnnamedAddr::No, fn_ty);
        self.intrinsics.borrow_mut().insert(name, (fn_ty, f));
        (fn_ty, f)
    }

    fn declare_intrinsic(&self, key: &str) -> Option<(&'ll Type, &'ll Value)> {
        macro_rules! ifn {
            ($name:expr, fn() -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, Some(&[]), $ret));
                }
            );
            ($name:expr, fn(...) -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, None, $ret));
                }
            );
            ($name:expr, fn($($arg:expr),*) -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, Some(&[$($arg),*]), $ret));
                }
            );
        }
        macro_rules! mk_struct {
            ($($field_ty:expr),*) => (self.type_struct( &[$($field_ty),*], false))
        }

        let ptr = self.type_ptr();
        let void = self.type_void();
        let i1 = self.type_i1();
        let t_i8 = self.type_i8();
        let t_i16 = self.type_i16();
        let t_i32 = self.type_i32();
        let t_i64 = self.type_i64();
        let t_i128 = self.type_i128();
        let t_isize = self.type_isize();
        let t_f16 = self.type_f16();
        let t_f32 = self.type_f32();
        let t_f64 = self.type_f64();
        let t_f128 = self.type_f128();
        let t_metadata = self.type_metadata();
        let t_token = self.type_token();

        ifn!("llvm.wasm.get.exception", fn(t_token) -> ptr);
        ifn!("llvm.wasm.get.ehselector", fn(t_token) -> t_i32);

        ifn!("llvm.wasm.trunc.unsigned.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.unsigned.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.unsigned.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.unsigned.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.wasm.trunc.signed.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.signed.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.signed.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.signed.i64.f64", fn(t_f64) -> t_i64);

        ifn!("llvm.fptosi.sat.i8.f32", fn(t_f32) -> t_i8);
        ifn!("llvm.fptosi.sat.i16.f32", fn(t_f32) -> t_i16);
        ifn!("llvm.fptosi.sat.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.fptosi.sat.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.fptosi.sat.i128.f32", fn(t_f32) -> t_i128);
        ifn!("llvm.fptosi.sat.i8.f64", fn(t_f64) -> t_i8);
        ifn!("llvm.fptosi.sat.i16.f64", fn(t_f64) -> t_i16);
        ifn!("llvm.fptosi.sat.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.fptosi.sat.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.fptosi.sat.i128.f64", fn(t_f64) -> t_i128);

        ifn!("llvm.fptoui.sat.i8.f32", fn(t_f32) -> t_i8);
        ifn!("llvm.fptoui.sat.i16.f32", fn(t_f32) -> t_i16);
        ifn!("llvm.fptoui.sat.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.fptoui.sat.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.fptoui.sat.i128.f32", fn(t_f32) -> t_i128);
        ifn!("llvm.fptoui.sat.i8.f64", fn(t_f64) -> t_i8);
        ifn!("llvm.fptoui.sat.i16.f64", fn(t_f64) -> t_i16);
        ifn!("llvm.fptoui.sat.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.fptoui.sat.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.fptoui.sat.i128.f64", fn(t_f64) -> t_i128);

        ifn!("llvm.trap", fn() -> void);
        ifn!("llvm.debugtrap", fn() -> void);
        ifn!("llvm.frameaddress", fn(t_i32) -> ptr);

        ifn!("llvm.powi.f16.i32", fn(t_f16, t_i32) -> t_f16);
        ifn!("llvm.powi.f32.i32", fn(t_f32, t_i32) -> t_f32);
        ifn!("llvm.powi.f64.i32", fn(t_f64, t_i32) -> t_f64);
        ifn!("llvm.powi.f128.i32", fn(t_f128, t_i32) -> t_f128);

        ifn!("llvm.pow.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.pow.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.pow.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.pow.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.sqrt.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.sqrt.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sqrt.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sqrt.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.sin.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.sin.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sin.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sin.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.cos.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.cos.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.cos.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.cos.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.exp.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.exp.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.exp2.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.exp2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp2.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.log.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.log.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.log10.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.log10.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log10.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log10.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.log2.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.log2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log2.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.fma.f16", fn(t_f16, t_f16, t_f16) -> t_f16);
        ifn!("llvm.fma.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
        ifn!("llvm.fma.f64", fn(t_f64, t_f64, t_f64) -> t_f64);
        ifn!("llvm.fma.f128", fn(t_f128, t_f128, t_f128) -> t_f128);

        ifn!("llvm.fmuladd.f16", fn(t_f16, t_f16, t_f16) -> t_f16);
        ifn!("llvm.fmuladd.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
        ifn!("llvm.fmuladd.f64", fn(t_f64, t_f64, t_f64) -> t_f64);
        ifn!("llvm.fmuladd.f128", fn(t_f128, t_f128, t_f128) -> t_f128);

        ifn!("llvm.fabs.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.fabs.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.fabs.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.fabs.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.minnum.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.minnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.minnum.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.minnum.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.minimum.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.minimum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.minimum.f64", fn(t_f64, t_f64) -> t_f64);
        // There are issues on x86_64 and aarch64 with the f128 variant.
        //  - https://github.com/llvm/llvm-project/issues/139380
        //  - https://github.com/llvm/llvm-project/issues/139381
        // ifn!("llvm.minimum.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.maxnum.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.maxnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.maxnum.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.maxnum.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.maximum.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.maximum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.maximum.f64", fn(t_f64, t_f64) -> t_f64);
        // There are issues on x86_64 and aarch64 with the f128 variant.
        //  - https://github.com/llvm/llvm-project/issues/139380
        //  - https://github.com/llvm/llvm-project/issues/139381
        // ifn!("llvm.maximum.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.floor.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.floor.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.floor.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.floor.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.ceil.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.ceil.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.ceil.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.ceil.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.trunc.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.trunc.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.trunc.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.trunc.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.copysign.f16", fn(t_f16, t_f16) -> t_f16);
        ifn!("llvm.copysign.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.copysign.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.copysign.f128", fn(t_f128, t_f128) -> t_f128);

        ifn!("llvm.round.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.round.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.round.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.round.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.roundeven.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.roundeven.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.roundeven.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.roundeven.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.rint.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.rint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.rint.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.rint.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.nearbyint.f16", fn(t_f16) -> t_f16);
        ifn!("llvm.nearbyint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.nearbyint.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.nearbyint.f128", fn(t_f128) -> t_f128);

        ifn!("llvm.ctpop.i8", fn(t_i8) -> t_i8);
        ifn!("llvm.ctpop.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.ctpop.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.ctpop.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.ctpop.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.ctlz.i8", fn(t_i8, i1) -> t_i8);
        ifn!("llvm.ctlz.i16", fn(t_i16, i1) -> t_i16);
        ifn!("llvm.ctlz.i32", fn(t_i32, i1) -> t_i32);
        ifn!("llvm.ctlz.i64", fn(t_i64, i1) -> t_i64);
        ifn!("llvm.ctlz.i128", fn(t_i128, i1) -> t_i128);

        ifn!("llvm.cttz.i8", fn(t_i8, i1) -> t_i8);
        ifn!("llvm.cttz.i16", fn(t_i16, i1) -> t_i16);
        ifn!("llvm.cttz.i32", fn(t_i32, i1) -> t_i32);
        ifn!("llvm.cttz.i64", fn(t_i64, i1) -> t_i64);
        ifn!("llvm.cttz.i128", fn(t_i128, i1) -> t_i128);

        ifn!("llvm.bswap.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.bswap.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.bswap.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.bswap.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.bitreverse.i8", fn(t_i8) -> t_i8);
        ifn!("llvm.bitreverse.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.bitreverse.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.bitreverse.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.bitreverse.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.fshl.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!("llvm.fshl.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!("llvm.fshl.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!("llvm.fshl.i64", fn(t_i64, t_i64, t_i64) -> t_i64);
        ifn!("llvm.fshl.i128", fn(t_i128, t_i128, t_i128) -> t_i128);

        ifn!("llvm.fshr.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!("llvm.fshr.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!("llvm.fshr.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!("llvm.fshr.i64", fn(t_i64, t_i64, t_i64) -> t_i64);
        ifn!("llvm.fshr.i128", fn(t_i128, t_i128, t_i128) -> t_i128);

        ifn!("llvm.sadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.sadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.sadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.sadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.sadd.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.uadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.uadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.uadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.uadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.uadd.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.ssub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.ssub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.ssub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.ssub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.ssub.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.usub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.usub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.usub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.usub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.usub.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.smul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.smul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.smul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.smul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.smul.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.umul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct! {t_i8, i1});
        ifn!("llvm.umul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct! {t_i16, i1});
        ifn!("llvm.umul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct! {t_i32, i1});
        ifn!("llvm.umul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct! {t_i64, i1});
        ifn!("llvm.umul.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct! {t_i128, i1});

        ifn!("llvm.sadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.sadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.sadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.sadd.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.sadd.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.uadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.uadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.uadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.uadd.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.uadd.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.ssub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.ssub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.ssub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.ssub.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.ssub.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.usub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.usub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.usub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.usub.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.usub.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.scmp.i8.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.scmp.i8.i16", fn(t_i16, t_i16) -> t_i8);
        ifn!("llvm.scmp.i8.i32", fn(t_i32, t_i32) -> t_i8);
        ifn!("llvm.scmp.i8.i64", fn(t_i64, t_i64) -> t_i8);
        ifn!("llvm.scmp.i8.i128", fn(t_i128, t_i128) -> t_i8);

        ifn!("llvm.ucmp.i8.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.ucmp.i8.i16", fn(t_i16, t_i16) -> t_i8);
        ifn!("llvm.ucmp.i8.i32", fn(t_i32, t_i32) -> t_i8);
        ifn!("llvm.ucmp.i8.i64", fn(t_i64, t_i64) -> t_i8);
        ifn!("llvm.ucmp.i8.i128", fn(t_i128, t_i128) -> t_i8);

        ifn!("llvm.lifetime.start.p0i8", fn(t_i64, ptr) -> void);
        ifn!("llvm.lifetime.end.p0i8", fn(t_i64, ptr) -> void);

        // FIXME: This is an infinitesimally small portion of the types you can
        // pass to this intrinsic, if we can ever lazily register intrinsics we
        // should register these when they're used, that way any type can be
        // passed.
        ifn!("llvm.is.constant.i1", fn(i1) -> i1);
        ifn!("llvm.is.constant.i8", fn(t_i8) -> i1);
        ifn!("llvm.is.constant.i16", fn(t_i16) -> i1);
        ifn!("llvm.is.constant.i32", fn(t_i32) -> i1);
        ifn!("llvm.is.constant.i64", fn(t_i64) -> i1);
        ifn!("llvm.is.constant.i128", fn(t_i128) -> i1);
        ifn!("llvm.is.constant.isize", fn(t_isize) -> i1);
        ifn!("llvm.is.constant.f16", fn(t_f16) -> i1);
        ifn!("llvm.is.constant.f32", fn(t_f32) -> i1);
        ifn!("llvm.is.constant.f64", fn(t_f64) -> i1);
        ifn!("llvm.is.constant.f128", fn(t_f128) -> i1);
        ifn!("llvm.is.constant.ptr", fn(ptr) -> i1);

        ifn!("llvm.expect.i1", fn(i1, i1) -> i1);
        ifn!("llvm.eh.typeid.for", fn(ptr) -> t_i32);
        ifn!("llvm.localescape", fn(...) -> void);
        ifn!("llvm.localrecover", fn(ptr, ptr, t_i32) -> ptr);
        ifn!("llvm.x86.seh.recoverfp", fn(ptr, ptr) -> ptr);

        ifn!("llvm.assume", fn(i1) -> void);
        ifn!("llvm.prefetch", fn(ptr, t_i32, t_i32, t_i32) -> void);

        // This isn't an "LLVM intrinsic", but LLVM's optimization passes
        // recognize it like one (including turning it into `bcmp` sometimes)
        // and we use it to implement intrinsics like `raw_eq` and `compare_bytes`
        match self.sess().target.arch.as_ref() {
            "avr" | "msp430" => ifn!("memcmp", fn(ptr, ptr, t_isize) -> t_i16),
            _ => ifn!("memcmp", fn(ptr, ptr, t_isize) -> t_i32),
        }

        // variadic intrinsics
        ifn!("llvm.va_start", fn(ptr) -> void);
        ifn!("llvm.va_end", fn(ptr) -> void);
        ifn!("llvm.va_copy", fn(ptr, ptr) -> void);

        if self.sess().instrument_coverage() {
            ifn!("llvm.instrprof.increment", fn(ptr, t_i64, t_i32, t_i32) -> void);
            ifn!("llvm.instrprof.mcdc.parameters", fn(ptr, t_i64, t_i32) -> void);
            ifn!("llvm.instrprof.mcdc.tvbitmap.update", fn(ptr, t_i64, t_i32, ptr) -> void);
        }

        ifn!("llvm.type.test", fn(ptr, t_metadata) -> i1);
        ifn!("llvm.type.checked.load", fn(ptr, t_i32, t_metadata) -> mk_struct! {ptr, i1});

        if self.sess().opts.debuginfo != DebugInfo::None {
            ifn!("llvm.dbg.declare", fn(t_metadata, t_metadata) -> void);
            ifn!("llvm.dbg.value", fn(t_metadata, t_i64, t_metadata) -> void);
        }

        ifn!("llvm.ptrmask", fn(ptr, t_isize) -> ptr);

        None
    }

    pub(crate) fn eh_catch_typeinfo(&self) -> &'ll Value {
        if let Some(eh_catch_typeinfo) = self.eh_catch_typeinfo.get() {
            return eh_catch_typeinfo;
        }
        let tcx = self.tcx;
        assert!(self.sess().target.os == "emscripten");
        let eh_catch_typeinfo = match tcx.lang_items().eh_catch_typeinfo() {
            Some(def_id) => self.get_static(def_id),
            _ => {
                let ty = self.type_struct(&[self.type_ptr(), self.type_ptr()], false);
                self.declare_global(&mangle_internal_symbol(self.tcx, "rust_eh_catch_typeinfo"), ty)
            }
        };
        self.eh_catch_typeinfo.set(Some(eh_catch_typeinfo));
        eh_catch_typeinfo
    }
}

impl CodegenCx<'_, '_> {
    /// Generates a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub(crate) fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push('.');
        name.push_str(&(idx as u64).to_base(ALPHANUMERIC_ONLY));
        name
    }
}

impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    /// A wrapper for [`llvm::LLVMSetMetadata`], but it takes `Metadata` as a parameter instead of `Value`.
    pub(crate) fn set_metadata<'a>(
        &self,
        val: &'a Value,
        kind_id: impl Into<llvm::MetadataKindId>,
        md: &'ll Metadata,
    ) {
        let node = self.get_metadata_value(md);
        llvm::LLVMSetMetadata(val, kind_id.into(), node);
    }
}

impl HasDataLayout for CodegenCx<'_, '_> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl HasTargetSpec for CodegenCx<'_, '_> {
    #[inline]
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for CodegenCx<'_, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, 'll> HasTypingEnv<'tcx> for CodegenCx<'ll, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'_, 'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_) | LayoutError::ReferencesError(_) = err {
            self.tcx.dcx().emit_fatal(Spanned { span, node: err.into_diagnostic() })
        } else {
            self.tcx.dcx().emit_fatal(ssa_errors::FailedToGetLayout { span, ty, err })
        }
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'_, 'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        match err {
            FnAbiError::Layout(LayoutError::SizeOverflow(_) | LayoutError::Cycle(_)) => {
                self.tcx.dcx().emit_fatal(Spanned { span, node: err });
            }
            _ => match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(span, "`fn_abi_of_fn_ptr({sig}, {extra_args:?})` failed: {err:?}",);
                }
                FnAbiRequest::OfInstance { instance, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({instance}, {extra_args:?})` failed: {err:?}",
                    );
                }
            },
        }
    }
}
