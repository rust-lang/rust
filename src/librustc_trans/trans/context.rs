// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm;
use llvm::{ContextRef, ModuleRef, ValueRef, BuilderRef};
use metadata::common::LinkMeta;
use middle::def::ExportMap;
use middle::def_id::DefId;
use middle::traits;
use trans::adt;
use trans::base;
use trans::builder::Builder;
use trans::common::{ExternMap,BuilderRef_res};
use trans::debuginfo;
use trans::declare;
use trans::glue::DropGlueKind;
use trans::monomorphize::MonoId;
use trans::type_::{Type, TypeNames};
use middle::subst::Substs;
use middle::ty::{self, Ty};
use session::config::NoDebugInfo;
use session::Session;
use util::sha2::Sha256;
use util::nodemap::{NodeMap, NodeSet, DefIdMap, FnvHashMap, FnvHashSet};

use std::ffi::CString;
use std::cell::{Cell, RefCell};
use std::ptr;
use std::rc::Rc;
use syntax::ast;
use syntax::parse::token::InternedString;

pub struct Stats {
    pub n_glues_created: Cell<usize>,
    pub n_null_glues: Cell<usize>,
    pub n_real_glues: Cell<usize>,
    pub n_fns: Cell<usize>,
    pub n_monos: Cell<usize>,
    pub n_inlines: Cell<usize>,
    pub n_closures: Cell<usize>,
    pub n_llvm_insns: Cell<usize>,
    pub llvm_insns: RefCell<FnvHashMap<String, usize>>,
    // (ident, llvm-instructions)
    pub fn_stats: RefCell<Vec<(String, usize)> >,
}

/// The shared portion of a `CrateContext`.  There is one `SharedCrateContext`
/// per crate.  The data here is shared between all compilation units of the
/// crate, so it must not contain references to any LLVM data structures
/// (aside from metadata-related ones).
pub struct SharedCrateContext<'a, 'tcx: 'a> {
    local_ccxs: Vec<LocalCrateContext<'tcx>>,

    metadata_llmod: ModuleRef,
    metadata_llcx: ContextRef,

    export_map: ExportMap,
    reachable: NodeSet,
    item_symbols: RefCell<NodeMap<String>>,
    link_meta: LinkMeta,
    symbol_hasher: RefCell<Sha256>,
    tcx: &'a ty::ctxt<'tcx>,
    stats: Stats,
    check_overflow: bool,
    check_drop_flag_for_sanity: bool,

    available_drop_glues: RefCell<FnvHashMap<DropGlueKind<'tcx>, String>>,
    use_dll_storage_attrs: bool,
}

/// The local portion of a `CrateContext`.  There is one `LocalCrateContext`
/// per compilation unit.  Each one has its own LLVM `ContextRef` so that
/// several compilation units may be optimized in parallel.  All other LLVM
/// data structures in the `LocalCrateContext` are tied to that `ContextRef`.
pub struct LocalCrateContext<'tcx> {
    llmod: ModuleRef,
    llcx: ContextRef,
    tn: TypeNames,
    externs: RefCell<ExternMap>,
    item_vals: RefCell<NodeMap<ValueRef>>,
    needs_unwind_cleanup_cache: RefCell<FnvHashMap<Ty<'tcx>, bool>>,
    fn_pointer_shims: RefCell<FnvHashMap<Ty<'tcx>, ValueRef>>,
    drop_glues: RefCell<FnvHashMap<DropGlueKind<'tcx>, ValueRef>>,
    /// Track mapping of external ids to local items imported for inlining
    external: RefCell<DefIdMap<Option<ast::NodeId>>>,
    /// Backwards version of the `external` map (inlined items to where they
    /// came from)
    external_srcs: RefCell<NodeMap<DefId>>,
    /// Cache instances of monomorphized functions
    monomorphized: RefCell<FnvHashMap<MonoId<'tcx>, ValueRef>>,
    monomorphizing: RefCell<DefIdMap<usize>>,
    available_monomorphizations: RefCell<FnvHashSet<String>>,
    /// Cache generated vtables
    vtables: RefCell<FnvHashMap<ty::PolyTraitRef<'tcx>, ValueRef>>,
    /// Cache of constant strings,
    const_cstr_cache: RefCell<FnvHashMap<InternedString, ValueRef>>,

    /// Reverse-direction for const ptrs cast from globals.
    /// Key is a ValueRef holding a *T,
    /// Val is a ValueRef holding a *[T].
    ///
    /// Needed because LLVM loses pointer->pointee association
    /// when we ptrcast, and we have to ptrcast during translation
    /// of a [T] const because we form a slice, a (*T,usize) pair, not
    /// a pointer to an LLVM array type. Similar for trait objects.
    const_unsized: RefCell<FnvHashMap<ValueRef, ValueRef>>,

    /// Cache of emitted const globals (value -> global)
    const_globals: RefCell<FnvHashMap<ValueRef, ValueRef>>,

    /// Cache of emitted const values
    const_values: RefCell<FnvHashMap<(ast::NodeId, &'tcx Substs<'tcx>), ValueRef>>,

    /// Cache of external const values
    extern_const_values: RefCell<DefIdMap<ValueRef>>,

    impl_method_cache: RefCell<FnvHashMap<(DefId, ast::Name), DefId>>,

    /// Cache of closure wrappers for bare fn's.
    closure_bare_wrapper_cache: RefCell<FnvHashMap<ValueRef, ValueRef>>,

    /// List of globals for static variables which need to be passed to the
    /// LLVM function ReplaceAllUsesWith (RAUW) when translation is complete.
    /// (We have to make sure we don't invalidate any ValueRefs referring
    /// to constants.)
    statics_to_rauw: RefCell<Vec<(ValueRef, ValueRef)>>,

    lltypes: RefCell<FnvHashMap<Ty<'tcx>, Type>>,
    llsizingtypes: RefCell<FnvHashMap<Ty<'tcx>, Type>>,
    adt_reprs: RefCell<FnvHashMap<Ty<'tcx>, Rc<adt::Repr<'tcx>>>>,
    type_hashcodes: RefCell<FnvHashMap<Ty<'tcx>, String>>,
    int_type: Type,
    opaque_vec_type: Type,
    builder: BuilderRef_res,

    /// Holds the LLVM values for closure IDs.
    closure_vals: RefCell<FnvHashMap<MonoId<'tcx>, ValueRef>>,

    dbg_cx: Option<debuginfo::CrateDebugContext<'tcx>>,

    eh_personality: RefCell<Option<ValueRef>>,
    rust_try_fn: RefCell<Option<ValueRef>>,
    unwind_resume_hooked: Cell<bool>,

    intrinsics: RefCell<FnvHashMap<&'static str, ValueRef>>,

    /// Number of LLVM instructions translated into this `LocalCrateContext`.
    /// This is used to perform some basic load-balancing to keep all LLVM
    /// contexts around the same size.
    n_llvm_insns: Cell<usize>,

    /// Depth of the current type-of computation - used to bail out
    type_of_depth: Cell<usize>,

    trait_cache: RefCell<FnvHashMap<ty::PolyTraitRef<'tcx>,
                                    traits::Vtable<'tcx, ()>>>,
}

pub struct CrateContext<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    local: &'a LocalCrateContext<'tcx>,
    /// The index of `local` in `shared.local_ccxs`.  This is used in
    /// `maybe_iter(true)` to identify the original `LocalCrateContext`.
    index: usize,
}

pub struct CrateContextIterator<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    index: usize,
}

impl<'a, 'tcx> Iterator for CrateContextIterator<'a,'tcx> {
    type Item = CrateContext<'a, 'tcx>;

    fn next(&mut self) -> Option<CrateContext<'a, 'tcx>> {
        if self.index >= self.shared.local_ccxs.len() {
            return None;
        }

        let index = self.index;
        self.index += 1;

        Some(CrateContext {
            shared: self.shared,
            local: &self.shared.local_ccxs[index],
            index: index,
        })
    }
}

/// The iterator produced by `CrateContext::maybe_iter`.
pub struct CrateContextMaybeIterator<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    index: usize,
    single: bool,
    origin: usize,
}

impl<'a, 'tcx> Iterator for CrateContextMaybeIterator<'a, 'tcx> {
    type Item = (CrateContext<'a, 'tcx>, bool);

    fn next(&mut self) -> Option<(CrateContext<'a, 'tcx>, bool)> {
        if self.index >= self.shared.local_ccxs.len() {
            return None;
        }

        let index = self.index;
        self.index += 1;
        if self.single {
            self.index = self.shared.local_ccxs.len();
        }

        let ccx = CrateContext {
            shared: self.shared,
            local: &self.shared.local_ccxs[index],
            index: index,
        };
        Some((ccx, index == self.origin))
    }
}


unsafe fn create_context_and_module(sess: &Session, mod_name: &str) -> (ContextRef, ModuleRef) {
    let llcx = llvm::LLVMContextCreate();
    let mod_name = CString::new(mod_name).unwrap();
    let llmod = llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx);

    if let Some(ref custom_data_layout) = sess.target.target.options.data_layout {
        let data_layout = CString::new(&custom_data_layout[..]).unwrap();
        llvm::LLVMSetDataLayout(llmod, data_layout.as_ptr());
    } else {
        let tm = ::back::write::create_target_machine(sess);
        llvm::LLVMRustSetDataLayoutFromTargetMachine(llmod, tm);
        llvm::LLVMRustDisposeTargetMachine(tm);
    }

    let llvm_target = sess.target.target.llvm_target.as_bytes();
    let llvm_target = CString::new(llvm_target).unwrap();
    llvm::LLVMRustSetNormalizedTarget(llmod, llvm_target.as_ptr());
    (llcx, llmod)
}

impl<'b, 'tcx> SharedCrateContext<'b, 'tcx> {
    pub fn new(crate_name: &str,
               local_count: usize,
               tcx: &'b ty::ctxt<'tcx>,
               export_map: ExportMap,
               symbol_hasher: Sha256,
               link_meta: LinkMeta,
               reachable: NodeSet,
               check_overflow: bool,
               check_drop_flag_for_sanity: bool)
               -> SharedCrateContext<'b, 'tcx> {
        let (metadata_llcx, metadata_llmod) = unsafe {
            create_context_and_module(&tcx.sess, "metadata")
        };

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
        // attribute to all LLVM functions that are reachable (e.g. they're
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

        let mut shared_ccx = SharedCrateContext {
            local_ccxs: Vec::with_capacity(local_count),
            metadata_llmod: metadata_llmod,
            metadata_llcx: metadata_llcx,
            export_map: export_map,
            reachable: reachable,
            item_symbols: RefCell::new(NodeMap()),
            link_meta: link_meta,
            symbol_hasher: RefCell::new(symbol_hasher),
            tcx: tcx,
            stats: Stats {
                n_glues_created: Cell::new(0),
                n_null_glues: Cell::new(0),
                n_real_glues: Cell::new(0),
                n_fns: Cell::new(0),
                n_monos: Cell::new(0),
                n_inlines: Cell::new(0),
                n_closures: Cell::new(0),
                n_llvm_insns: Cell::new(0),
                llvm_insns: RefCell::new(FnvHashMap()),
                fn_stats: RefCell::new(Vec::new()),
            },
            check_overflow: check_overflow,
            check_drop_flag_for_sanity: check_drop_flag_for_sanity,
            available_drop_glues: RefCell::new(FnvHashMap()),
            use_dll_storage_attrs: use_dll_storage_attrs,
        };

        for i in 0..local_count {
            // Append ".rs" to crate name as LLVM module identifier.
            //
            // LLVM code generator emits a ".file filename" directive
            // for ELF backends. Value of the "filename" is set as the
            // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
            // crashes if the module identifier is same as other symbols
            // such as a function name in the module.
            // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
            let llmod_id = format!("{}.{}.rs", crate_name, i);
            let local_ccx = LocalCrateContext::new(&shared_ccx, &llmod_id[..]);
            shared_ccx.local_ccxs.push(local_ccx);
        }

        shared_ccx
    }

    pub fn iter<'a>(&'a self) -> CrateContextIterator<'a, 'tcx> {
        CrateContextIterator {
            shared: self,
            index: 0,
        }
    }

    pub fn get_ccx<'a>(&'a self, index: usize) -> CrateContext<'a, 'tcx> {
        CrateContext {
            shared: self,
            local: &self.local_ccxs[index],
            index: index,
        }
    }

    fn get_smallest_ccx<'a>(&'a self) -> CrateContext<'a, 'tcx> {
        let (local_ccx, index) =
            self.local_ccxs
                .iter()
                .zip(0..self.local_ccxs.len())
                .min_by(|&(local_ccx, _idx)| local_ccx.n_llvm_insns.get())
                .unwrap();
        CrateContext {
            shared: self,
            local: local_ccx,
            index: index,
        }
    }


    pub fn metadata_llmod(&self) -> ModuleRef {
        self.metadata_llmod
    }

    pub fn metadata_llcx(&self) -> ContextRef {
        self.metadata_llcx
    }

    pub fn export_map<'a>(&'a self) -> &'a ExportMap {
        &self.export_map
    }

    pub fn reachable<'a>(&'a self) -> &'a NodeSet {
        &self.reachable
    }

    pub fn item_symbols<'a>(&'a self) -> &'a RefCell<NodeMap<String>> {
        &self.item_symbols
    }

    pub fn link_meta<'a>(&'a self) -> &'a LinkMeta {
        &self.link_meta
    }

    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.tcx
    }

    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    pub fn stats<'a>(&'a self) -> &'a Stats {
        &self.stats
    }

    pub fn use_dll_storage_attrs(&self) -> bool {
        self.use_dll_storage_attrs
    }
}

impl<'tcx> LocalCrateContext<'tcx> {
    fn new<'a>(shared: &SharedCrateContext<'a, 'tcx>,
           name: &str)
           -> LocalCrateContext<'tcx> {
        unsafe {
            let (llcx, llmod) = create_context_and_module(&shared.tcx.sess, name);

            let dbg_cx = if shared.tcx.sess.opts.debuginfo != NoDebugInfo {
                Some(debuginfo::CrateDebugContext::new(llmod))
            } else {
                None
            };

            let mut local_ccx = LocalCrateContext {
                llmod: llmod,
                llcx: llcx,
                tn: TypeNames::new(),
                externs: RefCell::new(FnvHashMap()),
                item_vals: RefCell::new(NodeMap()),
                needs_unwind_cleanup_cache: RefCell::new(FnvHashMap()),
                fn_pointer_shims: RefCell::new(FnvHashMap()),
                drop_glues: RefCell::new(FnvHashMap()),
                external: RefCell::new(DefIdMap()),
                external_srcs: RefCell::new(NodeMap()),
                monomorphized: RefCell::new(FnvHashMap()),
                monomorphizing: RefCell::new(DefIdMap()),
                available_monomorphizations: RefCell::new(FnvHashSet()),
                vtables: RefCell::new(FnvHashMap()),
                const_cstr_cache: RefCell::new(FnvHashMap()),
                const_unsized: RefCell::new(FnvHashMap()),
                const_globals: RefCell::new(FnvHashMap()),
                const_values: RefCell::new(FnvHashMap()),
                extern_const_values: RefCell::new(DefIdMap()),
                impl_method_cache: RefCell::new(FnvHashMap()),
                closure_bare_wrapper_cache: RefCell::new(FnvHashMap()),
                statics_to_rauw: RefCell::new(Vec::new()),
                lltypes: RefCell::new(FnvHashMap()),
                llsizingtypes: RefCell::new(FnvHashMap()),
                adt_reprs: RefCell::new(FnvHashMap()),
                type_hashcodes: RefCell::new(FnvHashMap()),
                int_type: Type::from_ref(ptr::null_mut()),
                opaque_vec_type: Type::from_ref(ptr::null_mut()),
                builder: BuilderRef_res(llvm::LLVMCreateBuilderInContext(llcx)),
                closure_vals: RefCell::new(FnvHashMap()),
                dbg_cx: dbg_cx,
                eh_personality: RefCell::new(None),
                rust_try_fn: RefCell::new(None),
                unwind_resume_hooked: Cell::new(false),
                intrinsics: RefCell::new(FnvHashMap()),
                n_llvm_insns: Cell::new(0),
                type_of_depth: Cell::new(0),
                trait_cache: RefCell::new(FnvHashMap()),
            };

            local_ccx.int_type = Type::int(&local_ccx.dummy_ccx(shared));
            local_ccx.opaque_vec_type = Type::opaque_vec(&local_ccx.dummy_ccx(shared));

            // Done mutating local_ccx directly.  (The rest of the
            // initialization goes through RefCell.)
            {
                let ccx = local_ccx.dummy_ccx(shared);

                let mut str_slice_ty = Type::named_struct(&ccx, "str_slice");
                str_slice_ty.set_struct_body(&[Type::i8p(&ccx), ccx.int_type()], false);
                ccx.tn().associate_type("str_slice", &str_slice_ty);

                if ccx.sess().count_llvm_insns() {
                    base::init_insn_ctxt()
                }
            }

            local_ccx
        }
    }

    /// Create a dummy `CrateContext` from `self` and  the provided
    /// `SharedCrateContext`.  This is somewhat dangerous because `self` may
    /// not actually be an element of `shared.local_ccxs`, which can cause some
    /// operations to panic unexpectedly.
    ///
    /// This is used in the `LocalCrateContext` constructor to allow calling
    /// functions that expect a complete `CrateContext`, even before the local
    /// portion is fully initialized and attached to the `SharedCrateContext`.
    fn dummy_ccx<'a>(&'a self, shared: &'a SharedCrateContext<'a, 'tcx>)
                     -> CrateContext<'a, 'tcx> {
        CrateContext {
            shared: shared,
            local: self,
            index: !0 as usize,
        }
    }
}

impl<'b, 'tcx> CrateContext<'b, 'tcx> {
    pub fn shared(&self) -> &'b SharedCrateContext<'b, 'tcx> {
        self.shared
    }

    pub fn local(&self) -> &'b LocalCrateContext<'tcx> {
        self.local
    }


    /// Get a (possibly) different `CrateContext` from the same
    /// `SharedCrateContext`.
    pub fn rotate(&self) -> CrateContext<'b, 'tcx> {
        self.shared.get_smallest_ccx()
    }

    /// Either iterate over only `self`, or iterate over all `CrateContext`s in
    /// the `SharedCrateContext`.  The iterator produces `(ccx, is_origin)`
    /// pairs, where `is_origin` is `true` if `ccx` is `self` and `false`
    /// otherwise.  This method is useful for avoiding code duplication in
    /// cases where it may or may not be necessary to translate code into every
    /// context.
    pub fn maybe_iter(&self, iter_all: bool) -> CrateContextMaybeIterator<'b, 'tcx> {
        CrateContextMaybeIterator {
            shared: self.shared,
            index: if iter_all { 0 } else { self.index },
            single: !iter_all,
            origin: self.index,
        }
    }


    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.shared.tcx
    }

    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.shared.tcx.sess
    }

    pub fn builder<'a>(&'a self) -> Builder<'a, 'tcx> {
        Builder::new(self)
    }

    pub fn raw_builder<'a>(&'a self) -> BuilderRef {
        self.local.builder.b
    }

    pub fn get_intrinsic(&self, key: &str) -> ValueRef {
        if let Some(v) = self.intrinsics().borrow().get(key).cloned() {
            return v;
        }
        match declare_intrinsic(self, key) {
            Some(v) => return v,
            None => panic!("unknown intrinsic '{}'", key)
        }
    }

    pub fn llmod(&self) -> ModuleRef {
        self.local.llmod
    }

    pub fn llcx(&self) -> ContextRef {
        self.local.llcx
    }

    pub fn td(&self) -> llvm::TargetDataRef {
        unsafe { llvm::LLVMRustGetModuleDataLayout(self.llmod()) }
    }

    pub fn tn<'a>(&'a self) -> &'a TypeNames {
        &self.local.tn
    }

    pub fn externs<'a>(&'a self) -> &'a RefCell<ExternMap> {
        &self.local.externs
    }

    pub fn item_vals<'a>(&'a self) -> &'a RefCell<NodeMap<ValueRef>> {
        &self.local.item_vals
    }

    pub fn export_map<'a>(&'a self) -> &'a ExportMap {
        &self.shared.export_map
    }

    pub fn reachable<'a>(&'a self) -> &'a NodeSet {
        &self.shared.reachable
    }

    pub fn item_symbols<'a>(&'a self) -> &'a RefCell<NodeMap<String>> {
        &self.shared.item_symbols
    }

    pub fn link_meta<'a>(&'a self) -> &'a LinkMeta {
        &self.shared.link_meta
    }

    pub fn needs_unwind_cleanup_cache(&self) -> &RefCell<FnvHashMap<Ty<'tcx>, bool>> {
        &self.local.needs_unwind_cleanup_cache
    }

    pub fn fn_pointer_shims(&self) -> &RefCell<FnvHashMap<Ty<'tcx>, ValueRef>> {
        &self.local.fn_pointer_shims
    }

    pub fn drop_glues<'a>(&'a self) -> &'a RefCell<FnvHashMap<DropGlueKind<'tcx>, ValueRef>> {
        &self.local.drop_glues
    }

    pub fn external<'a>(&'a self) -> &'a RefCell<DefIdMap<Option<ast::NodeId>>> {
        &self.local.external
    }

    pub fn external_srcs<'a>(&'a self) -> &'a RefCell<NodeMap<DefId>> {
        &self.local.external_srcs
    }

    pub fn monomorphized<'a>(&'a self) -> &'a RefCell<FnvHashMap<MonoId<'tcx>, ValueRef>> {
        &self.local.monomorphized
    }

    pub fn monomorphizing<'a>(&'a self) -> &'a RefCell<DefIdMap<usize>> {
        &self.local.monomorphizing
    }

    pub fn vtables<'a>(&'a self) -> &'a RefCell<FnvHashMap<ty::PolyTraitRef<'tcx>, ValueRef>> {
        &self.local.vtables
    }

    pub fn const_cstr_cache<'a>(&'a self) -> &'a RefCell<FnvHashMap<InternedString, ValueRef>> {
        &self.local.const_cstr_cache
    }

    pub fn const_unsized<'a>(&'a self) -> &'a RefCell<FnvHashMap<ValueRef, ValueRef>> {
        &self.local.const_unsized
    }

    pub fn const_globals<'a>(&'a self) -> &'a RefCell<FnvHashMap<ValueRef, ValueRef>> {
        &self.local.const_globals
    }

    pub fn const_values<'a>(&'a self) -> &'a RefCell<FnvHashMap<(ast::NodeId, &'tcx Substs<'tcx>),
                                                                ValueRef>> {
        &self.local.const_values
    }

    pub fn extern_const_values<'a>(&'a self) -> &'a RefCell<DefIdMap<ValueRef>> {
        &self.local.extern_const_values
    }

    pub fn impl_method_cache<'a>(&'a self)
            -> &'a RefCell<FnvHashMap<(DefId, ast::Name), DefId>> {
        &self.local.impl_method_cache
    }

    pub fn closure_bare_wrapper_cache<'a>(&'a self) -> &'a RefCell<FnvHashMap<ValueRef, ValueRef>> {
        &self.local.closure_bare_wrapper_cache
    }

    pub fn statics_to_rauw<'a>(&'a self) -> &'a RefCell<Vec<(ValueRef, ValueRef)>> {
        &self.local.statics_to_rauw
    }

    pub fn lltypes<'a>(&'a self) -> &'a RefCell<FnvHashMap<Ty<'tcx>, Type>> {
        &self.local.lltypes
    }

    pub fn llsizingtypes<'a>(&'a self) -> &'a RefCell<FnvHashMap<Ty<'tcx>, Type>> {
        &self.local.llsizingtypes
    }

    pub fn adt_reprs<'a>(&'a self) -> &'a RefCell<FnvHashMap<Ty<'tcx>, Rc<adt::Repr<'tcx>>>> {
        &self.local.adt_reprs
    }

    pub fn symbol_hasher<'a>(&'a self) -> &'a RefCell<Sha256> {
        &self.shared.symbol_hasher
    }

    pub fn type_hashcodes<'a>(&'a self) -> &'a RefCell<FnvHashMap<Ty<'tcx>, String>> {
        &self.local.type_hashcodes
    }

    pub fn stats<'a>(&'a self) -> &'a Stats {
        &self.shared.stats
    }

    pub fn available_monomorphizations<'a>(&'a self) -> &'a RefCell<FnvHashSet<String>> {
        &self.local.available_monomorphizations
    }

    pub fn available_drop_glues(&self) -> &RefCell<FnvHashMap<DropGlueKind<'tcx>, String>> {
        &self.shared.available_drop_glues
    }

    pub fn int_type(&self) -> Type {
        self.local.int_type
    }

    pub fn opaque_vec_type(&self) -> Type {
        self.local.opaque_vec_type
    }

    pub fn closure_vals<'a>(&'a self) -> &'a RefCell<FnvHashMap<MonoId<'tcx>, ValueRef>> {
        &self.local.closure_vals
    }

    pub fn dbg_cx<'a>(&'a self) -> &'a Option<debuginfo::CrateDebugContext<'tcx>> {
        &self.local.dbg_cx
    }

    pub fn eh_personality<'a>(&'a self) -> &'a RefCell<Option<ValueRef>> {
        &self.local.eh_personality
    }

    pub fn rust_try_fn<'a>(&'a self) -> &'a RefCell<Option<ValueRef>> {
        &self.local.rust_try_fn
    }

    pub fn unwind_resume_hooked<'a>(&'a self) -> &'a Cell<bool> {
        &self.local.unwind_resume_hooked
    }

    fn intrinsics<'a>(&'a self) -> &'a RefCell<FnvHashMap<&'static str, ValueRef>> {
        &self.local.intrinsics
    }

    pub fn count_llvm_insn(&self) {
        self.local.n_llvm_insns.set(self.local.n_llvm_insns.get() + 1);
    }

    pub fn trait_cache(&self) -> &RefCell<FnvHashMap<ty::PolyTraitRef<'tcx>,
                                                     traits::Vtable<'tcx, ()>>> {
        &self.local.trait_cache
    }

    /// Return exclusive upper bound on object size.
    ///
    /// The theoretical maximum object size is defined as the maximum positive `int` value. This
    /// ensures that the `offset` semantics remain well-defined by allowing it to correctly index
    /// every address within an object along with one byte past the end, along with allowing `int`
    /// to store the difference between any two pointers into an object.
    ///
    /// The upper bound on 64-bit currently needs to be lower because LLVM uses a 64-bit integer to
    /// represent object size in bits. It would need to be 1 << 61 to account for this, but is
    /// currently conservatively bounded to 1 << 47 as that is enough to cover the current usable
    /// address space on 64-bit ARMv8 and x86_64.
    pub fn obj_size_bound(&self) -> u64 {
        match &self.sess().target.target.target_pointer_width[..] {
            "32" => 1 << 31,
            "64" => 1 << 47,
            _ => unreachable!() // error handled by config::build_target_config
        }
    }

    pub fn report_overbig_object(&self, obj: Ty<'tcx>) -> ! {
        self.sess().fatal(
            &format!("the type `{:?}` is too big for the current architecture",
                    obj))
    }

    pub fn enter_type_of(&self, ty: Ty<'tcx>) -> TypeOfDepthLock<'b, 'tcx> {
        let current_depth = self.local.type_of_depth.get();
        debug!("enter_type_of({:?}) at depth {:?}", ty, current_depth);
        if current_depth > self.sess().recursion_limit.get() {
            self.sess().fatal(
                &format!("overflow representing the type `{}`", ty))
        }
        self.local.type_of_depth.set(current_depth + 1);
        TypeOfDepthLock(self.local)
    }

    pub fn check_overflow(&self) -> bool {
        self.shared.check_overflow
    }

    pub fn check_drop_flag_for_sanity(&self) -> bool {
        // This controls whether we emit a conditional llvm.debugtrap
        // guarded on whether the dropflag is one of its (two) valid
        // values.
        self.shared.check_drop_flag_for_sanity
    }

    pub fn use_dll_storage_attrs(&self) -> bool {
        self.shared.use_dll_storage_attrs()
    }
}

pub struct TypeOfDepthLock<'a, 'tcx: 'a>(&'a LocalCrateContext<'tcx>);

impl<'a, 'tcx> Drop for TypeOfDepthLock<'a, 'tcx> {
    fn drop(&mut self) {
        self.0.type_of_depth.set(self.0.type_of_depth.get() - 1);
    }
}

/// Declare any llvm intrinsics that you might need
fn declare_intrinsic(ccx: &CrateContext, key: &str) -> Option<ValueRef> {
    macro_rules! ifn {
        ($name:expr, fn() -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(ccx, $name, Type::func(&[], &$ret),
                                             ccx.tcx().mk_nil());
                llvm::SetUnnamedAddr(f, false);
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, fn($($arg:expr),*) -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(ccx, $name, Type::func(&[$($arg),*], &$ret),
                                             ccx.tcx().mk_nil());
                llvm::SetUnnamedAddr(f, false);
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        )
    }
    macro_rules! mk_struct {
        ($($field_ty:expr),*) => (Type::struct_(ccx, &[$($field_ty),*], false))
    }

    let i8p = Type::i8p(ccx);
    let void = Type::void(ccx);
    let i1 = Type::i1(ccx);
    let t_i8 = Type::i8(ccx);
    let t_i16 = Type::i16(ccx);
    let t_i32 = Type::i32(ccx);
    let t_i64 = Type::i64(ccx);
    let t_f32 = Type::f32(ccx);
    let t_f64 = Type::f64(ccx);

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

    ifn!("llvm.ctlz.i8", fn(t_i8 , i1) -> t_i8);
    ifn!("llvm.ctlz.i16", fn(t_i16, i1) -> t_i16);
    ifn!("llvm.ctlz.i32", fn(t_i32, i1) -> t_i32);
    ifn!("llvm.ctlz.i64", fn(t_i64, i1) -> t_i64);

    ifn!("llvm.cttz.i8", fn(t_i8 , i1) -> t_i8);
    ifn!("llvm.cttz.i16", fn(t_i16, i1) -> t_i16);
    ifn!("llvm.cttz.i32", fn(t_i32, i1) -> t_i32);
    ifn!("llvm.cttz.i64", fn(t_i64, i1) -> t_i64);

    ifn!("llvm.bswap.i16", fn(t_i16) -> t_i16);
    ifn!("llvm.bswap.i32", fn(t_i32) -> t_i32);
    ifn!("llvm.bswap.i64", fn(t_i64) -> t_i64);

    ifn!("llvm.sadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.sadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.sadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.sadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.uadd.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.uadd.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.uadd.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.uadd.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.ssub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.ssub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.ssub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.ssub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.usub.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.usub.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.usub.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.usub.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.smul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.smul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.smul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.smul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.umul.with.overflow.i8", fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.umul.with.overflow.i16", fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.umul.with.overflow.i32", fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.umul.with.overflow.i64", fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.lifetime.start", fn(t_i64,i8p) -> void);
    ifn!("llvm.lifetime.end", fn(t_i64, i8p) -> void);

    ifn!("llvm.expect.i1", fn(i1, i1) -> i1);
    ifn!("llvm.eh.typeid.for", fn(i8p) -> t_i32);

    // Some intrinsics were introduced in later versions of LLVM, but they have
    // fallbacks in libc or libm and such.
    macro_rules! compatible_ifn {
        ($name:expr, noop($cname:ident ($($arg:expr),*) -> void), $llvm_version:expr) => (
            if unsafe { llvm::LLVMVersionMinor() >= $llvm_version } {
                // The `if key == $name` is already in ifn!
                ifn!($name, fn($($arg),*) -> void);
            } else if key == $name {
                let f = declare::declare_cfn(ccx, stringify!($cname),
                                             Type::func(&[$($arg),*], &void),
                                             ccx.tcx().mk_nil());
                llvm::SetLinkage(f, llvm::InternalLinkage);

                let bld = ccx.builder();
                let llbb = unsafe {
                    llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), f,
                                                        "entry-block\0".as_ptr() as *const _)
                };

                bld.position_at_end(llbb);
                bld.ret_void();

                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, $cname:ident ($($arg:expr),*) -> $ret:expr, $llvm_version:expr) => (
            if unsafe { llvm::LLVMVersionMinor() >= $llvm_version } {
                // The `if key == $name` is already in ifn!
                ifn!($name, fn($($arg),*) -> $ret);
            } else if key == $name {
                let f = declare::declare_cfn(ccx, stringify!($cname),
                                             Type::func(&[$($arg),*], &$ret),
                                             ccx.tcx().mk_nil());
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        )
    }

    compatible_ifn!("llvm.assume", noop(llvmcompat_assume(i1) -> void), 6);

    if ccx.sess().opts.debuginfo != NoDebugInfo {
        ifn!("llvm.dbg.declare", fn(Type::metadata(ccx), Type::metadata(ccx)) -> void);
        ifn!("llvm.dbg.value", fn(Type::metadata(ccx), t_i64, Type::metadata(ccx)) -> void);
    }
    return None;
}
