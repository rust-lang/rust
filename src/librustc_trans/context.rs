// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common;
use llvm;
use llvm::{ContextRef, ModuleRef, ValueRef};
use rustc::dep_graph::DepGraphSafe;
use rustc::hir;
use rustc::hir::def_id::DefId;
use debuginfo;
use callee;
use base;
use declare;
use monomorphize::Instance;

use monomorphize::partitioning::CodegenUnit;
use type_::Type;
use type_of::PointeeInfo;

use rustc_data_structures::base_n;
use rustc::mir::mono::Stats;
use rustc::session::config::{self, NoDebugInfo};
use rustc::session::Session;
use rustc::ty::layout::{LayoutError, LayoutOf, Size, TyLayout};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::util::nodemap::FxHashMap;

use std::ffi::{CStr, CString};
use std::cell::{Cell, RefCell};
use std::ptr;
use std::iter;
use std::str;
use std::sync::Arc;
use syntax::symbol::InternedString;
use abi::Abi;

/// There is one `CodegenCx` per compilation unit. Each one has its own LLVM
/// `ContextRef` so that several compilation units may be optimized in parallel.
/// All other LLVM data structures in the `CodegenCx` are tied to that `ContextRef`.
pub struct CodegenCx<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub check_overflow: bool,
    pub use_dll_storage_attrs: bool,
    pub tls_model: llvm::ThreadLocalMode,

    pub llmod: ModuleRef,
    pub llcx: ContextRef,
    pub stats: RefCell<Stats>,
    pub codegen_unit: Arc<CodegenUnit<'tcx>>,

    /// Cache instances of monomorphic and polymorphic items
    pub instances: RefCell<FxHashMap<Instance<'tcx>, ValueRef>>,
    /// Cache generated vtables
    pub vtables: RefCell<FxHashMap<(Ty<'tcx>,
                                Option<ty::PolyExistentialTraitRef<'tcx>>), ValueRef>>,
    /// Cache of constant strings,
    pub const_cstr_cache: RefCell<FxHashMap<InternedString, ValueRef>>,

    /// Reverse-direction for const ptrs cast from globals.
    /// Key is a ValueRef holding a *T,
    /// Val is a ValueRef holding a *[T].
    ///
    /// Needed because LLVM loses pointer->pointee association
    /// when we ptrcast, and we have to ptrcast during translation
    /// of a [T] const because we form a slice, a (*T,usize) pair, not
    /// a pointer to an LLVM array type. Similar for trait objects.
    pub const_unsized: RefCell<FxHashMap<ValueRef, ValueRef>>,

    /// Cache of emitted const globals (value -> global)
    pub const_globals: RefCell<FxHashMap<ValueRef, ValueRef>>,

    /// Mapping from static definitions to their DefId's.
    pub statics: RefCell<FxHashMap<ValueRef, DefId>>,

    /// List of globals for static variables which need to be passed to the
    /// LLVM function ReplaceAllUsesWith (RAUW) when translation is complete.
    /// (We have to make sure we don't invalidate any ValueRefs referring
    /// to constants.)
    pub statics_to_rauw: RefCell<Vec<(ValueRef, ValueRef)>>,

    /// Statics that will be placed in the llvm.used variable
    /// See http://llvm.org/docs/LangRef.html#the-llvm-used-global-variable for details
    pub used_statics: RefCell<Vec<ValueRef>>,

    pub lltypes: RefCell<FxHashMap<(Ty<'tcx>, Option<usize>), Type>>,
    pub scalar_lltypes: RefCell<FxHashMap<Ty<'tcx>, Type>>,
    pub pointee_infos: RefCell<FxHashMap<(Ty<'tcx>, Size), Option<PointeeInfo>>>,
    pub isize_ty: Type,

    pub dbg_cx: Option<debuginfo::CrateDebugContext<'tcx>>,

    eh_personality: Cell<Option<ValueRef>>,
    eh_unwind_resume: Cell<Option<ValueRef>>,
    pub rust_try_fn: Cell<Option<ValueRef>>,

    intrinsics: RefCell<FxHashMap<&'static str, ValueRef>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,
}

impl<'a, 'tcx> DepGraphSafe for CodegenCx<'a, 'tcx> {
}

pub fn get_reloc_model(sess: &Session) -> llvm::RelocMode {
    let reloc_model_arg = match sess.opts.cg.relocation_model {
        Some(ref s) => &s[..],
        None => &sess.target.target.options.relocation_model[..],
    };

    match ::back::write::RELOC_MODEL_ARGS.iter().find(
        |&&arg| arg.0 == reloc_model_arg) {
        Some(x) => x.1,
        _ => {
            sess.err(&format!("{:?} is not a valid relocation mode",
                              reloc_model_arg));
            sess.abort_if_errors();
            bug!();
        }
    }
}

fn get_tls_model(sess: &Session) -> llvm::ThreadLocalMode {
    let tls_model_arg = match sess.opts.debugging_opts.tls_model {
        Some(ref s) => &s[..],
        None => &sess.target.target.options.tls_model[..],
    };

    match ::back::write::TLS_MODEL_ARGS.iter().find(
        |&&arg| arg.0 == tls_model_arg) {
        Some(x) => x.1,
        _ => {
            sess.err(&format!("{:?} is not a valid TLS model",
                              tls_model_arg));
            sess.abort_if_errors();
            bug!();
        }
    }
}

fn is_any_library(sess: &Session) -> bool {
    sess.crate_types.borrow().iter().any(|ty| {
        *ty != config::CrateTypeExecutable
    })
}

pub fn is_pie_binary(sess: &Session) -> bool {
    !is_any_library(sess) && get_reloc_model(sess) == llvm::RelocMode::PIC
}

pub unsafe fn create_context_and_module(sess: &Session, mod_name: &str) -> (ContextRef, ModuleRef) {
    let llcx = llvm::LLVMRustContextCreate(sess.fewer_names());
    let mod_name = CString::new(mod_name).unwrap();
    let llmod = llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx);

    // Ensure the data-layout values hardcoded remain the defaults.
    if sess.target.target.options.is_builtin {
        let tm = ::back::write::create_target_machine(sess, false);
        llvm::LLVMRustSetDataLayoutFromTargetMachine(llmod, tm);
        llvm::LLVMRustDisposeTargetMachine(tm);

        let data_layout = llvm::LLVMGetDataLayout(llmod);
        let data_layout = str::from_utf8(CStr::from_ptr(data_layout).to_bytes())
            .ok().expect("got a non-UTF8 data-layout from LLVM");

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

        if !custom_llvm_used && sess.target.target.data_layout != data_layout {
            bug!("data-layout for builtin `{}` target, `{}`, \
                  differs from LLVM default, `{}`",
                 sess.target.target.llvm_target,
                 sess.target.target.data_layout,
                 data_layout);
        }
    }

    let data_layout = CString::new(&sess.target.target.data_layout[..]).unwrap();
    llvm::LLVMSetDataLayout(llmod, data_layout.as_ptr());

    let llvm_target = sess.target.target.llvm_target.as_bytes();
    let llvm_target = CString::new(llvm_target).unwrap();
    llvm::LLVMRustSetNormalizedTarget(llmod, llvm_target.as_ptr());

    if is_pie_binary(sess) {
        llvm::LLVMRustSetModulePIELevel(llmod);
    }

    (llcx, llmod)
}

impl<'a, 'tcx> CodegenCx<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               codegen_unit: Arc<CodegenUnit<'tcx>>,
               llmod_id: &str)
               -> CodegenCx<'a, 'tcx> {
        // An interesting part of Windows which MSVC forces our hand on (and
        // apparently MinGW didn't) is the usage of `dllimport` and `dllexport`
        // attributes in LLVM IR as well as native dependencies (in C these
        // correspond to `__declspec(dllimport)`).
        //
        // Whenever a dynamic library is built by MSVC it must have its public
        // interface specified by functions tagged with `dllexport` or otherwise
        // they're not available to be linked against. This poses a few problems
        // for the compiler, some of which are somewhat fundamental, but we use
        // the `use_dll_storage_attrs` variable below to attach the `dllexport`
        // attribute to all LLVM functions that are exported e.g. they're
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
        // start) and then strongly recommending static linkage on MSVC!
        let use_dll_storage_attrs = tcx.sess.target.target.options.is_like_msvc;

        let check_overflow = tcx.sess.overflow_checks();

        let tls_model = get_tls_model(&tcx.sess);

        unsafe {
            let (llcx, llmod) = create_context_and_module(&tcx.sess,
                                                          &llmod_id[..]);

            let dbg_cx = if tcx.sess.opts.debuginfo != NoDebugInfo {
                let dctx = debuginfo::CrateDebugContext::new(llmod);
                debuginfo::metadata::compile_unit_metadata(tcx,
                                                           codegen_unit.name(),
                                                           &dctx);
                Some(dctx)
            } else {
                None
            };

            let mut cx = CodegenCx {
                tcx,
                check_overflow,
                use_dll_storage_attrs,
                tls_model,
                llmod,
                llcx,
                stats: RefCell::new(Stats::default()),
                codegen_unit,
                instances: RefCell::new(FxHashMap()),
                vtables: RefCell::new(FxHashMap()),
                const_cstr_cache: RefCell::new(FxHashMap()),
                const_unsized: RefCell::new(FxHashMap()),
                const_globals: RefCell::new(FxHashMap()),
                statics: RefCell::new(FxHashMap()),
                statics_to_rauw: RefCell::new(Vec::new()),
                used_statics: RefCell::new(Vec::new()),
                lltypes: RefCell::new(FxHashMap()),
                scalar_lltypes: RefCell::new(FxHashMap()),
                pointee_infos: RefCell::new(FxHashMap()),
                isize_ty: Type::from_ref(ptr::null_mut()),
                dbg_cx,
                eh_personality: Cell::new(None),
                eh_unwind_resume: Cell::new(None),
                rust_try_fn: Cell::new(None),
                intrinsics: RefCell::new(FxHashMap()),
                local_gen_sym_counter: Cell::new(0),
            };
            cx.isize_ty = Type::isize(&cx);
            cx
        }
    }

    pub fn into_stats(self) -> Stats {
        self.stats.into_inner()
    }
}

impl<'b, 'tcx> CodegenCx<'b, 'tcx> {
    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    pub fn get_intrinsic(&self, key: &str) -> ValueRef {
        if let Some(v) = self.intrinsics.borrow().get(key).cloned() {
            return v;
        }
        match declare_intrinsic(self, key) {
            Some(v) => return v,
            None => bug!("unknown intrinsic '{}'", key)
        }
    }

    /// Generate a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push_str(".");
        base_n::push_str(idx as u128, base_n::ALPHANUMERIC_ONLY, &mut name);
        name
    }

    pub fn eh_personality(&self) -> ValueRef {
        // The exception handling personality function.
        //
        // If our compilation unit has the `eh_personality` lang item somewhere
        // within it, then we just need to translate that. Otherwise, we're
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
            return llpersonality
        }
        let tcx = self.tcx;
        let llfn = match tcx.lang_items().eh_personality() {
            Some(def_id) if !base::wants_msvc_seh(self.sess()) => {
                callee::resolve_and_get_fn(self, def_id, tcx.intern_substs(&[]))
            }
            _ => {
                let name = if base::wants_msvc_seh(self.sess()) {
                    "__CxxFrameHandler3"
                } else {
                    "rust_eh_personality"
                };
                let fty = Type::variadic_func(&[], &Type::i32(self));
                declare::declare_cfn(self, name, fty)
            }
        };
        self.eh_personality.set(Some(llfn));
        llfn
    }

    // Returns a ValueRef of the "eh_unwind_resume" lang item if one is defined,
    // otherwise declares it as an external function.
    pub fn eh_unwind_resume(&self) -> ValueRef {
        use attributes;
        let unwresume = &self.eh_unwind_resume;
        if let Some(llfn) = unwresume.get() {
            return llfn;
        }

        let tcx = self.tcx;
        assert!(self.sess().target.target.options.custom_unwind_resume);
        if let Some(def_id) = tcx.lang_items().eh_unwind_resume() {
            let llfn = callee::resolve_and_get_fn(self, def_id, tcx.intern_substs(&[]));
            unwresume.set(Some(llfn));
            return llfn;
        }

        let ty = tcx.mk_fn_ptr(ty::Binder(tcx.mk_fn_sig(
            iter::once(tcx.mk_mut_ptr(tcx.types.u8)),
            tcx.types.never,
            false,
            hir::Unsafety::Unsafe,
            Abi::C
        )));

        let llfn = declare::declare_fn(self, "rust_eh_unwind_resume", ty);
        attributes::unwind(llfn, true);
        unwresume.set(Some(llfn));
        llfn
    }

    pub fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        common::type_needs_drop(self.tcx, ty)
    }

    pub fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        common::type_is_sized(self.tcx, ty)
    }

    pub fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        common::type_is_freeze(self.tcx, ty)
    }

    pub fn type_has_metadata(&self, ty: Ty<'tcx>) -> bool {
        use syntax_pos::DUMMY_SP;
        if ty.is_sized(self.tcx.at(DUMMY_SP), ty::ParamEnv::reveal_all()) {
            return false;
        }

        let tail = self.tcx.struct_tail(ty);
        match tail.sty {
            ty::TyForeign(..) => false,
            ty::TyStr | ty::TySlice(..) | ty::TyDynamic(..) => true,
            _ => bug!("unexpected unsized tail: {:?}", tail.sty),
        }
    }
}

impl<'a, 'tcx> ty::layout::HasDataLayout for &'a CodegenCx<'a, 'tcx> {
    fn data_layout(&self) -> &ty::layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'tcx> ty::layout::HasTyCtxt<'tcx> for &'a CodegenCx<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> {
        self.tcx
    }
}

impl<'a, 'tcx> LayoutOf<Ty<'tcx>> for &'a CodegenCx<'a, 'tcx> {
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty))
            .unwrap_or_else(|e| match e {
                LayoutError::SizeOverflow(_) => self.sess().fatal(&e.to_string()),
                _ => bug!("failed to get layout for `{}`: {}", ty, e)
            })
    }
}

/// Declare any llvm intrinsics that you might need
fn declare_intrinsic(cx: &CodegenCx, key: &str) -> Option<ValueRef> {
    macro_rules! ifn {
        ($name:expr, fn() -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(cx, $name, Type::func(&[], &$ret));
                llvm::SetUnnamedAddr(f, false);
                cx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, fn(...) -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(cx, $name, Type::variadic_func(&[], &$ret));
                llvm::SetUnnamedAddr(f, false);
                cx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, fn($($arg:expr),*) -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(cx, $name, Type::func(&[$($arg),*], &$ret));
                llvm::SetUnnamedAddr(f, false);
                cx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
    }
    macro_rules! mk_struct {
        ($($field_ty:expr),*) => (Type::struct_(cx, &[$($field_ty),*], false))
    }

    let i8p = Type::i8p(cx);
    let void = Type::void(cx);
    let i1 = Type::i1(cx);
    let t_i8 = Type::i8(cx);
    let t_i16 = Type::i16(cx);
    let t_i32 = Type::i32(cx);
    let t_i64 = Type::i64(cx);
    let t_i128 = Type::i128(cx);
    let t_f32 = Type::f32(cx);
    let t_f64 = Type::f64(cx);

    ifn!("llvm.memcpy.p0i8.p0i8.i16", fn(i8p, i8p, t_i16, t_i32, i1) -> void);
    ifn!("llvm.memcpy.p0i8.p0i8.i32", fn(i8p, i8p, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memcpy.p0i8.p0i8.i64", fn(i8p, i8p, t_i64, t_i32, i1) -> void);
    ifn!("llvm.memmove.p0i8.p0i8.i16", fn(i8p, i8p, t_i16, t_i32, i1) -> void);
    ifn!("llvm.memmove.p0i8.p0i8.i32", fn(i8p, i8p, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memmove.p0i8.p0i8.i64", fn(i8p, i8p, t_i64, t_i32, i1) -> void);
    ifn!("llvm.memset.p0i8.i16", fn(i8p, t_i8, t_i16, t_i32, i1) -> void);
    ifn!("llvm.memset.p0i8.i32", fn(i8p, t_i8, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memset.p0i8.i64", fn(i8p, t_i8, t_i64, t_i32, i1) -> void);

    ifn!("llvm.trap", fn() -> void);
    ifn!("llvm.debugtrap", fn() -> void);
    ifn!("llvm.frameaddress", fn(t_i32) -> i8p);

    ifn!("llvm.powi.f32", fn(t_f32, t_i32) -> t_f32);
    ifn!("llvm.powi.f64", fn(t_f64, t_i32) -> t_f64);
    ifn!("llvm.pow.f32", fn(t_f32, t_f32) -> t_f32);
    ifn!("llvm.pow.f64", fn(t_f64, t_f64) -> t_f64);

    ifn!("llvm.sqrt.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.sqrt.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.sin.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.sin.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.cos.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.cos.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.exp.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.exp.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.exp2.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.exp2.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.log.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.log.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.log10.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.log10.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.log2.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.log2.f64", fn(t_f64) -> t_f64);

    ifn!("llvm.fma.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
    ifn!("llvm.fma.f64", fn(t_f64, t_f64, t_f64) -> t_f64);

    ifn!("llvm.fabs.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.fabs.f64", fn(t_f64) -> t_f64);

    ifn!("llvm.floor.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.floor.f64", fn(t_f64) -> t_f64);
    ifn!("llvm.ceil.f32", fn(t_f32) -> t_f32);
    ifn!("llvm.ceil.f64", fn(t_f64) -> t_f64);
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

    ifn!("llvm.ctlz.i8", fn(t_i8 , i1) -> t_i8);
    ifn!("llvm.ctlz.i16", fn(t_i16, i1) -> t_i16);
    ifn!("llvm.ctlz.i32", fn(t_i32, i1) -> t_i32);
    ifn!("llvm.ctlz.i64", fn(t_i64, i1) -> t_i64);
    ifn!("llvm.ctlz.i128", fn(t_i128, i1) -> t_i128);

    ifn!("llvm.cttz.i8", fn(t_i8 , i1) -> t_i8);
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

    ifn!("llvm.sadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.sadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.sadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.sadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.sadd.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.uadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.uadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.uadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.uadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.uadd.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.ssub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.ssub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.ssub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.ssub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.ssub.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.usub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.usub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.usub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.usub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.usub.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.smul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.smul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.smul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.smul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.smul.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.umul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.umul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.umul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.umul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});
    ifn!("llvm.umul.with.overflow.i128", fn(t_i128, t_i128) -> mk_struct!{t_i128, i1});

    ifn!("llvm.lifetime.start", fn(t_i64,i8p) -> void);
    ifn!("llvm.lifetime.end", fn(t_i64, i8p) -> void);

    ifn!("llvm.expect.i1", fn(i1, i1) -> i1);
    ifn!("llvm.eh.typeid.for", fn(i8p) -> t_i32);
    ifn!("llvm.localescape", fn(...) -> void);
    ifn!("llvm.localrecover", fn(i8p, i8p, t_i32) -> i8p);
    ifn!("llvm.x86.seh.recoverfp", fn(i8p, i8p) -> i8p);

    ifn!("llvm.assume", fn(i1) -> void);
    ifn!("llvm.prefetch", fn(i8p, t_i32, t_i32, t_i32) -> void);

    if cx.sess().opts.debuginfo != NoDebugInfo {
        ifn!("llvm.dbg.declare", fn(Type::metadata(cx), Type::metadata(cx)) -> void);
        ifn!("llvm.dbg.value", fn(Type::metadata(cx), t_i64, Type::metadata(cx)) -> void);
    }
    return None;
}
