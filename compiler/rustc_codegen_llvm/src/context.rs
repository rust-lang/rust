use crate::attributes;
use crate::callee::get_fn;
use crate::coverageinfo;
use crate::debuginfo;
use crate::llvm;
use crate::llvm_util;
use crate::type_::Type;
use crate::value::Value;

use rustc_codegen_ssa::base::wants_msvc_seh;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::base_n;
use rustc_data_structures::const_cstr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::bug;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::layout::{HasParamEnv, LayoutError, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_session::config::{CFGuard, CrateType, DebugInfo};
use rustc_session::Session;
use rustc_span::source_map::{Span, DUMMY_SP};
use rustc_span::symbol::Symbol;
use rustc_target::abi::{HasDataLayout, LayoutOf, PointeeInfo, Size, TargetDataLayout, VariantIdx};
use rustc_target::spec::{HasTargetSpec, RelocModel, Target, TlsModel};

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::str;

/// There is one `CodegenCx` per compilation unit. Each one has its own LLVM
/// `llvm::Context` so that several compilation units may be optimized in parallel.
/// All other LLVM data structures in the `CodegenCx` are tied to that `llvm::Context`.
pub struct CodegenCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub check_overflow: bool,
    pub use_dll_storage_attrs: bool,
    pub tls_model: llvm::ThreadLocalMode,

    pub llmod: &'ll llvm::Module,
    pub llcx: &'ll llvm::Context,
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,

    /// Cache instances of monomorphic and polymorphic items
    pub instances: RefCell<FxHashMap<Instance<'tcx>, &'ll Value>>,
    /// Cache generated vtables
    pub vtables:
        RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), &'ll Value>>,
    /// Cache of constant strings,
    pub const_cstr_cache: RefCell<FxHashMap<Symbol, &'ll Value>>,

    /// Reverse-direction for const ptrs cast from globals.
    ///
    /// Key is a Value holding a `*T`,
    /// Val is a Value holding a `*[T]`.
    ///
    /// Needed because LLVM loses pointer->pointee association
    /// when we ptrcast, and we have to ptrcast during codegen
    /// of a `[T]` const because we form a slice, a `(*T,usize)` pair, not
    /// a pointer to an LLVM array type. Similar for trait objects.
    pub const_unsized: RefCell<FxHashMap<&'ll Value, &'ll Value>>,

    /// Cache of emitted const globals (value -> global)
    pub const_globals: RefCell<FxHashMap<&'ll Value, &'ll Value>>,

    /// List of globals for static variables which need to be passed to the
    /// LLVM function ReplaceAllUsesWith (RAUW) when codegen is complete.
    /// (We have to make sure we don't invalidate any Values referring
    /// to constants.)
    pub statics_to_rauw: RefCell<Vec<(&'ll Value, &'ll Value)>>,

    /// Statics that will be placed in the llvm.used variable
    /// See <http://llvm.org/docs/LangRef.html#the-llvm-used-global-variable> for details
    pub used_statics: RefCell<Vec<&'ll Value>>,

    pub lltypes: RefCell<FxHashMap<(Ty<'tcx>, Option<VariantIdx>), &'ll Type>>,
    pub scalar_lltypes: RefCell<FxHashMap<Ty<'tcx>, &'ll Type>>,
    pub pointee_infos: RefCell<FxHashMap<(Ty<'tcx>, Size), Option<PointeeInfo>>>,
    pub isize_ty: &'ll Type,

    pub coverage_cx: Option<coverageinfo::CrateCoverageContext<'tcx>>,
    pub dbg_cx: Option<debuginfo::CrateDebugContext<'ll, 'tcx>>,

    eh_personality: Cell<Option<&'ll Value>>,
    eh_catch_typeinfo: Cell<Option<&'ll Value>>,
    pub rust_try_fn: Cell<Option<&'ll Value>>,

    intrinsics: RefCell<FxHashMap<&'static str, &'ll Value>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,
}

fn to_llvm_tls_model(tls_model: TlsModel) -> llvm::ThreadLocalMode {
    match tls_model {
        TlsModel::GeneralDynamic => llvm::ThreadLocalMode::GeneralDynamic,
        TlsModel::LocalDynamic => llvm::ThreadLocalMode::LocalDynamic,
        TlsModel::InitialExec => llvm::ThreadLocalMode::InitialExec,
        TlsModel::LocalExec => llvm::ThreadLocalMode::LocalExec,
    }
}

fn strip_x86_address_spaces(data_layout: String) -> String {
    data_layout.replace("-p270:32:32-p271:32:32-p272:64:64-", "-")
}

pub unsafe fn create_module(
    tcx: TyCtxt<'_>,
    llcx: &'ll llvm::Context,
    mod_name: &str,
) -> &'ll llvm::Module {
    let sess = tcx.sess;
    let mod_name = SmallCStr::new(mod_name);
    let llmod = llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx);

    let mut target_data_layout = sess.target.data_layout.clone();
    if llvm_util::get_major_version() < 10
        && (sess.target.arch == "x86" || sess.target.arch == "x86_64")
    {
        target_data_layout = strip_x86_address_spaces(target_data_layout);
    }

    // Ensure the data-layout values hardcoded remain the defaults.
    if sess.target.is_builtin {
        let tm = crate::back::write::create_informational_target_machine(tcx.sess);
        llvm::LLVMRustSetDataLayoutFromTargetMachine(llmod, tm);
        llvm::LLVMRustDisposeTargetMachine(tm);

        let llvm_data_layout = llvm::LLVMGetDataLayoutStr(llmod);
        let llvm_data_layout = str::from_utf8(CStr::from_ptr(llvm_data_layout).to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

        // Unfortunately LLVM target specs change over time, and right now we
        // don't have proper support to work with any more than one
        // `data_layout` than the one that is in the rust-lang/rust repo. If
        // this compiler is configured against a custom LLVM, we may have a
        // differing data layout, even though we should update our own to use
        // that one.
        //
        // As an interim hack, if CFG_LLVM_ROOT is not an empty string then we
        // disable this check entirely as we may be configured with something
        // that has a different target layout.
        //
        // Unsure if this will actually cause breakage when rustc is configured
        // as such.
        //
        // FIXME(#34960)
        let cfg_llvm_root = option_env!("CFG_LLVM_ROOT").unwrap_or("");
        let custom_llvm_used = cfg_llvm_root.trim() != "";

        if !custom_llvm_used && target_data_layout != llvm_data_layout {
            bug!(
                "data-layout for builtin `{}` target, `{}`, \
                  differs from LLVM default, `{}`",
                sess.target.llvm_target,
                target_data_layout,
                llvm_data_layout
            );
        }
    }

    let data_layout = SmallCStr::new(&target_data_layout);
    llvm::LLVMSetDataLayout(llmod, data_layout.as_ptr());

    let llvm_target = SmallCStr::new(&sess.target.llvm_target);
    llvm::LLVMRustSetNormalizedTarget(llmod, llvm_target.as_ptr());

    if sess.relocation_model() == RelocModel::Pic {
        llvm::LLVMRustSetModulePICLevel(llmod);
        // PIE is potentially more effective than PIC, but can only be used in executables.
        // If all our outputs are executables, then we can relax PIC to PIE.
        if sess.crate_types().iter().all(|ty| *ty == CrateType::Executable) {
            llvm::LLVMRustSetModulePIELevel(llmod);
        }
    }

    // If skipping the PLT is enabled, we need to add some module metadata
    // to ensure intrinsic calls don't use it.
    if !sess.needs_plt() {
        let avoid_plt = "RtLibUseGOT\0".as_ptr().cast();
        llvm::LLVMRustAddModuleFlag(llmod, avoid_plt, 1);
    }

    // Control Flow Guard is currently only supported by the MSVC linker on Windows.
    if sess.target.is_like_msvc {
        match sess.opts.cg.control_flow_guard {
            CFGuard::Disabled => {}
            CFGuard::NoChecks => {
                // Set `cfguard=1` module flag to emit metadata only.
                llvm::LLVMRustAddModuleFlag(llmod, "cfguard\0".as_ptr() as *const _, 1)
            }
            CFGuard::Checks => {
                // Set `cfguard=2` module flag to emit metadata and checks.
                llvm::LLVMRustAddModuleFlag(llmod, "cfguard\0".as_ptr() as *const _, 2)
            }
        }
    }

    llmod
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    crate fn new(
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
        // The flipside of this situation is that whenever you link to a dll and
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

        let check_overflow = tcx.sess.overflow_checks();

        let tls_model = to_llvm_tls_model(tcx.sess.tls_model());

        let (llcx, llmod) = (&*llvm_module.llcx, llvm_module.llmod());

        let coverage_cx = if tcx.sess.opts.debugging_opts.instrument_coverage {
            let covctx = coverageinfo::CrateCoverageContext::new();
            Some(covctx)
        } else {
            None
        };

        let dbg_cx = if tcx.sess.opts.debuginfo != DebugInfo::None {
            let dctx = debuginfo::CrateDebugContext::new(llmod);
            debuginfo::metadata::compile_unit_metadata(tcx, &codegen_unit.name().as_str(), &dctx);
            Some(dctx)
        } else {
            None
        };

        let isize_ty = Type::ix_llcx(llcx, tcx.data_layout.pointer_size.bits());

        CodegenCx {
            tcx,
            check_overflow,
            use_dll_storage_attrs,
            tls_model,
            llmod,
            llcx,
            codegen_unit,
            instances: Default::default(),
            vtables: Default::default(),
            const_cstr_cache: Default::default(),
            const_unsized: Default::default(),
            const_globals: Default::default(),
            statics_to_rauw: RefCell::new(Vec::new()),
            used_statics: RefCell::new(Vec::new()),
            lltypes: Default::default(),
            scalar_lltypes: Default::default(),
            pointee_infos: Default::default(),
            isize_ty,
            coverage_cx,
            dbg_cx,
            eh_personality: Cell::new(None),
            eh_catch_typeinfo: Cell::new(None),
            rust_try_fn: Cell::new(None),
            intrinsics: Default::default(),
            local_gen_sym_counter: Cell::new(0),
        }
    }

    crate fn statics_to_rauw(&self) -> &RefCell<Vec<(&'ll Value, &'ll Value)>> {
        &self.statics_to_rauw
    }

    #[inline]
    pub fn coverage_context(&'a self) -> Option<&'a coverageinfo::CrateCoverageContext<'tcx>> {
        self.coverage_cx.as_ref()
    }
}

impl MiscMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), &'ll Value>>
    {
        &self.vtables
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
        let tcx = self.tcx;
        let llfn = match tcx.lang_items().eh_personality() {
            Some(def_id) if !wants_msvc_seh(self.sess()) => self.get_fn_addr(
                ty::Instance::resolve(
                    tcx,
                    ty::ParamEnv::reveal_all(),
                    def_id,
                    tcx.intern_substs(&[]),
                )
                .unwrap()
                .unwrap(),
            ),
            _ => {
                let name = if wants_msvc_seh(self.sess()) {
                    "__CxxFrameHandler3"
                } else {
                    "rust_eh_personality"
                };
                let fty = self.type_variadic_func(&[], self.type_i32());
                self.declare_cfn(name, fty)
            }
        };
        attributes::apply_target_cpu_attr(self, llfn);
        self.eh_personality.set(Some(llfn));
        llfn
    }

    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn check_overflow(&self) -> bool {
        self.check_overflow
    }

    fn codegen_unit(&self) -> &'tcx CodegenUnit<'tcx> {
        self.codegen_unit
    }

    fn used_statics(&self) -> &RefCell<Vec<&'ll Value>> {
        &self.used_statics
    }

    fn set_frame_pointer_elimination(&self, llfn: &'ll Value) {
        attributes::set_frame_pointer_elimination(self, llfn)
    }

    fn apply_target_cpu_attr(&self, llfn: &'ll Value) {
        attributes::apply_target_cpu_attr(self, llfn);
        attributes::apply_tune_cpu_attr(self, llfn);
    }

    fn create_used_variable(&self) {
        let name = const_cstr!("llvm.used");
        let section = const_cstr!("llvm.metadata");
        let array =
            self.const_array(&self.type_ptr_to(self.type_i8()), &*self.used_statics.borrow());

        unsafe {
            let g = llvm::LLVMAddGlobal(self.llmod, self.val_ty(array), name.as_ptr());
            llvm::LLVMSetInitializer(g, array);
            llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage);
            llvm::LLVMSetSection(g, section.as_ptr());
        }
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        if self.get_declared_value("main").is_none() {
            Some(self.declare_cfn("main", fn_type))
        } else {
            // If the symbol already exists, it is an error: for example, the user wrote
            // #[no_mangle] extern "C" fn main(..) {..}
            // instead of #[start]
            None
        }
    }
}

impl CodegenCx<'b, 'tcx> {
    crate fn get_intrinsic(&self, key: &str) -> &'b Value {
        if let Some(v) = self.intrinsics.borrow().get(key).cloned() {
            return v;
        }

        self.declare_intrinsic(key).unwrap_or_else(|| bug!("unknown intrinsic '{}'", key))
    }

    fn insert_intrinsic(
        &self,
        name: &'static str,
        args: Option<&[&'b llvm::Type]>,
        ret: &'b llvm::Type,
    ) -> &'b llvm::Value {
        let fn_ty = if let Some(args) = args {
            self.type_func(args, ret)
        } else {
            self.type_variadic_func(&[], ret)
        };
        let f = self.declare_cfn(name, fn_ty);
        llvm::SetUnnamedAddress(f, llvm::UnnamedAddr::No);
        self.intrinsics.borrow_mut().insert(name, f);
        f
    }

    fn declare_intrinsic(&self, key: &str) -> Option<&'b Value> {
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

        let i8p = self.type_i8p();
        let void = self.type_void();
        let i1 = self.type_i1();
        let t_i8 = self.type_i8();
        let t_i16 = self.type_i16();
        let t_i32 = self.type_i32();
        let t_i64 = self.type_i64();
        let t_i128 = self.type_i128();
        let t_f32 = self.type_f32();
        let t_f64 = self.type_f64();

        macro_rules! vector_types {
            ($id_out:ident: $elem_ty:ident, $len:expr) => {
                let $id_out = self.type_vector($elem_ty, $len);
            };
            ($($id_out:ident: $elem_ty:ident, $len:expr;)*) => {
                $(vector_types!($id_out: $elem_ty, $len);)*
            }
        }
        vector_types! {
            t_v2f32: t_f32, 2;
            t_v4f32: t_f32, 4;
            t_v8f32: t_f32, 8;
            t_v16f32: t_f32, 16;

            t_v2f64: t_f64, 2;
            t_v4f64: t_f64, 4;
            t_v8f64: t_f64, 8;
        }

        ifn!("llvm.wasm.trunc.saturate.unsigned.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.saturate.unsigned.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.saturate.unsigned.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.saturate.unsigned.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.wasm.trunc.saturate.signed.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.saturate.signed.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.saturate.signed.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.saturate.signed.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.wasm.trunc.unsigned.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.unsigned.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.unsigned.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.unsigned.i64.f64", fn(t_f64) -> t_i64);
        ifn!("llvm.wasm.trunc.signed.i32.f32", fn(t_f32) -> t_i32);
        ifn!("llvm.wasm.trunc.signed.i32.f64", fn(t_f64) -> t_i32);
        ifn!("llvm.wasm.trunc.signed.i64.f32", fn(t_f32) -> t_i64);
        ifn!("llvm.wasm.trunc.signed.i64.f64", fn(t_f64) -> t_i64);

        ifn!("llvm.trap", fn() -> void);
        ifn!("llvm.debugtrap", fn() -> void);
        ifn!("llvm.frameaddress", fn(t_i32) -> i8p);
        ifn!("llvm.sideeffect", fn() -> void);

        ifn!("llvm.powi.f32", fn(t_f32, t_i32) -> t_f32);
        ifn!("llvm.powi.v2f32", fn(t_v2f32, t_i32) -> t_v2f32);
        ifn!("llvm.powi.v4f32", fn(t_v4f32, t_i32) -> t_v4f32);
        ifn!("llvm.powi.v8f32", fn(t_v8f32, t_i32) -> t_v8f32);
        ifn!("llvm.powi.v16f32", fn(t_v16f32, t_i32) -> t_v16f32);
        ifn!("llvm.powi.f64", fn(t_f64, t_i32) -> t_f64);
        ifn!("llvm.powi.v2f64", fn(t_v2f64, t_i32) -> t_v2f64);
        ifn!("llvm.powi.v4f64", fn(t_v4f64, t_i32) -> t_v4f64);
        ifn!("llvm.powi.v8f64", fn(t_v8f64, t_i32) -> t_v8f64);

        ifn!("llvm.pow.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.pow.v2f32", fn(t_v2f32, t_v2f32) -> t_v2f32);
        ifn!("llvm.pow.v4f32", fn(t_v4f32, t_v4f32) -> t_v4f32);
        ifn!("llvm.pow.v8f32", fn(t_v8f32, t_v8f32) -> t_v8f32);
        ifn!("llvm.pow.v16f32", fn(t_v16f32, t_v16f32) -> t_v16f32);
        ifn!("llvm.pow.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.pow.v2f64", fn(t_v2f64, t_v2f64) -> t_v2f64);
        ifn!("llvm.pow.v4f64", fn(t_v4f64, t_v4f64) -> t_v4f64);
        ifn!("llvm.pow.v8f64", fn(t_v8f64, t_v8f64) -> t_v8f64);

        ifn!("llvm.sqrt.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sqrt.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.sqrt.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.sqrt.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.sqrt.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.sqrt.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sqrt.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.sqrt.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.sqrt.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.sin.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sin.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.sin.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.sin.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.sin.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.sin.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sin.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.sin.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.sin.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.cos.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.cos.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.cos.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.cos.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.cos.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.cos.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.cos.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.cos.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.cos.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.exp.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.exp.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.exp.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.exp.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.exp.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.exp.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.exp.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.exp2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp2.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.exp2.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.exp2.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.exp2.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.exp2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp2.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.exp2.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.exp2.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log10.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log10.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log10.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log10.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log10.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log10.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log10.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log10.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log10.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log2.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log2.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log2.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log2.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log2.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log2.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log2.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.fma.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
        ifn!("llvm.fma.v2f32", fn(t_v2f32, t_v2f32, t_v2f32) -> t_v2f32);
        ifn!("llvm.fma.v4f32", fn(t_v4f32, t_v4f32, t_v4f32) -> t_v4f32);
        ifn!("llvm.fma.v8f32", fn(t_v8f32, t_v8f32, t_v8f32) -> t_v8f32);
        ifn!("llvm.fma.v16f32", fn(t_v16f32, t_v16f32, t_v16f32) -> t_v16f32);
        ifn!("llvm.fma.f64", fn(t_f64, t_f64, t_f64) -> t_f64);
        ifn!("llvm.fma.v2f64", fn(t_v2f64, t_v2f64, t_v2f64) -> t_v2f64);
        ifn!("llvm.fma.v4f64", fn(t_v4f64, t_v4f64, t_v4f64) -> t_v4f64);
        ifn!("llvm.fma.v8f64", fn(t_v8f64, t_v8f64, t_v8f64) -> t_v8f64);

        ifn!("llvm.fabs.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.fabs.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.fabs.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.fabs.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.fabs.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.fabs.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.fabs.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.fabs.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.fabs.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.minnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.minnum.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.maxnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.maxnum.f64", fn(t_f64, t_f64) -> t_f64);

        ifn!("llvm.floor.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.floor.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.floor.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.floor.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.floor.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.floor.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.floor.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.floor.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.floor.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.ceil.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.ceil.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.ceil.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.ceil.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.ceil.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.ceil.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.ceil.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.ceil.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.ceil.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.trunc.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.trunc.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.copysign.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.copysign.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.round.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.round.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.rint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.rint.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.nearbyint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.nearbyint.f64", fn(t_f64) -> t_f64);

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

        ifn!("llvm.lifetime.start.p0i8", fn(t_i64, i8p) -> void);
        ifn!("llvm.lifetime.end.p0i8", fn(t_i64, i8p) -> void);

        ifn!("llvm.expect.i1", fn(i1, i1) -> i1);
        ifn!("llvm.eh.typeid.for", fn(i8p) -> t_i32);
        ifn!("llvm.localescape", fn(...) -> void);
        ifn!("llvm.localrecover", fn(i8p, i8p, t_i32) -> i8p);
        ifn!("llvm.x86.seh.recoverfp", fn(i8p, i8p) -> i8p);

        ifn!("llvm.assume", fn(i1) -> void);
        ifn!("llvm.prefetch", fn(i8p, t_i32, t_i32, t_i32) -> void);

        // variadic intrinsics
        ifn!("llvm.va_start", fn(i8p) -> void);
        ifn!("llvm.va_end", fn(i8p) -> void);
        ifn!("llvm.va_copy", fn(i8p, i8p) -> void);

        if self.sess().opts.debugging_opts.instrument_coverage {
            ifn!("llvm.instrprof.increment", fn(i8p, t_i64, t_i32, t_i32) -> void);
        }

        if self.sess().opts.debuginfo != DebugInfo::None {
            ifn!("llvm.dbg.declare", fn(self.type_metadata(), self.type_metadata()) -> void);
            ifn!("llvm.dbg.value", fn(self.type_metadata(), t_i64, self.type_metadata()) -> void);
        }
        None
    }

    crate fn eh_catch_typeinfo(&self) -> &'b Value {
        if let Some(eh_catch_typeinfo) = self.eh_catch_typeinfo.get() {
            return eh_catch_typeinfo;
        }
        let tcx = self.tcx;
        assert!(self.sess().target.is_like_emscripten);
        let eh_catch_typeinfo = match tcx.lang_items().eh_catch_typeinfo() {
            Some(def_id) => self.get_static(def_id),
            _ => {
                let ty = self
                    .type_struct(&[self.type_ptr_to(self.type_isize()), self.type_i8p()], false);
                self.declare_global("rust_eh_catch_typeinfo", ty)
            }
        };
        let eh_catch_typeinfo = self.const_bitcast(eh_catch_typeinfo, self.type_i8p());
        self.eh_catch_typeinfo.set(Some(eh_catch_typeinfo));
        eh_catch_typeinfo
    }
}

impl<'b, 'tcx> CodegenCx<'b, 'tcx> {
    /// Generates a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push('.');
        base_n::push_str(idx as u128, base_n::ALPHANUMERIC_ONLY, &mut name);
        name
    }
}

impl HasDataLayout for CodegenCx<'ll, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl HasTargetSpec for CodegenCx<'ll, 'tcx> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target
    }
}

impl ty::layout::HasTyCtxt<'tcx> for CodegenCx<'ll, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl LayoutOf for CodegenCx<'ll, 'tcx> {
    type Ty = Ty<'tcx>;
    type TyAndLayout = TyAndLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyAndLayout {
        self.spanned_layout_of(ty, DUMMY_SP)
    }

    fn spanned_layout_of(&self, ty: Ty<'tcx>, span: Span) -> Self::TyAndLayout {
        self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty)).unwrap_or_else(|e| {
            if let LayoutError::SizeOverflow(_) = e {
                self.sess().span_fatal(span, &e.to_string())
            } else {
                bug!("failed to get layout for `{}`: {}", ty, e)
            }
        })
    }
}

impl<'tcx, 'll> HasParamEnv<'tcx> for CodegenCx<'ll, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::reveal_all()
    }
}
