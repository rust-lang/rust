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
use llvm::{ContextRef, ModuleRef, ValueRef};
use rustc::dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig, WorkProduct};
use middle::cstore::LinkMeta;
use rustc::hir;
use rustc::hir::def::ExportMap;
use rustc::hir::def_id::DefId;
use rustc::traits;
use debuginfo;
use callee::Callee;
use base;
use declare;
use glue::DropGlueKind;
use monomorphize::Instance;

use partitioning::CodegenUnit;
use trans_item::TransItem;
use type_::Type;
use rustc_data_structures::base_n;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use session::config::NoDebugInfo;
use session::Session;
use session::config;
use symbol_map::SymbolMap;
use util::nodemap::{NodeSet, DefIdMap, FxHashMap, FxHashSet};

use std::ffi::{CStr, CString};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ptr;
use std::iter;
use std::rc::Rc;
use std::str;
use syntax::ast;
use syntax::symbol::InternedString;
use syntax_pos::DUMMY_SP;
use abi::{Abi, FnType};

pub struct Stats {
    pub n_glues_created: Cell<usize>,
    pub n_null_glues: Cell<usize>,
    pub n_real_glues: Cell<usize>,
    pub n_fns: Cell<usize>,
    pub n_inlines: Cell<usize>,
    pub n_closures: Cell<usize>,
    pub n_llvm_insns: Cell<usize>,
    pub llvm_insns: RefCell<FxHashMap<String, usize>>,
    // (ident, llvm-instructions)
    pub fn_stats: RefCell<Vec<(String, usize)> >,
}

/// The shared portion of a `CrateContext`.  There is one `SharedCrateContext`
/// per crate.  The data here is shared between all compilation units of the
/// crate, so it must not contain references to any LLVM data structures
/// (aside from metadata-related ones).
pub struct SharedCrateContext<'a, 'tcx: 'a> {
    metadata_llmod: ModuleRef,
    metadata_llcx: ContextRef,

    export_map: ExportMap,
    exported_symbols: NodeSet,
    link_meta: LinkMeta,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    empty_param_env: ty::ParameterEnvironment<'tcx>,
    stats: Stats,
    check_overflow: bool,

    use_dll_storage_attrs: bool,

    translation_items: RefCell<FxHashSet<TransItem<'tcx>>>,
    trait_cache: RefCell<DepTrackingMap<TraitSelectionCache<'tcx>>>,
    project_cache: RefCell<DepTrackingMap<ProjectionCache<'tcx>>>,
}

/// The local portion of a `CrateContext`.  There is one `LocalCrateContext`
/// per compilation unit.  Each one has its own LLVM `ContextRef` so that
/// several compilation units may be optimized in parallel.  All other LLVM
/// data structures in the `LocalCrateContext` are tied to that `ContextRef`.
pub struct LocalCrateContext<'tcx> {
    llmod: ModuleRef,
    llcx: ContextRef,
    previous_work_product: Option<WorkProduct>,
    codegen_unit: CodegenUnit<'tcx>,
    needs_unwind_cleanup_cache: RefCell<FxHashMap<Ty<'tcx>, bool>>,
    fn_pointer_shims: RefCell<FxHashMap<Ty<'tcx>, ValueRef>>,
    drop_glues: RefCell<FxHashMap<DropGlueKind<'tcx>, (ValueRef, FnType)>>,
    /// Cache instances of monomorphic and polymorphic items
    instances: RefCell<FxHashMap<Instance<'tcx>, ValueRef>>,
    /// Cache generated vtables
    vtables: RefCell<FxHashMap<(ty::Ty<'tcx>,
                                Option<ty::PolyExistentialTraitRef<'tcx>>), ValueRef>>,
    /// Cache of constant strings,
    const_cstr_cache: RefCell<FxHashMap<InternedString, ValueRef>>,

    /// Reverse-direction for const ptrs cast from globals.
    /// Key is a ValueRef holding a *T,
    /// Val is a ValueRef holding a *[T].
    ///
    /// Needed because LLVM loses pointer->pointee association
    /// when we ptrcast, and we have to ptrcast during translation
    /// of a [T] const because we form a slice, a (*T,usize) pair, not
    /// a pointer to an LLVM array type. Similar for trait objects.
    const_unsized: RefCell<FxHashMap<ValueRef, ValueRef>>,

    /// Cache of emitted const globals (value -> global)
    const_globals: RefCell<FxHashMap<ValueRef, ValueRef>>,

    /// Cache of emitted const values
    const_values: RefCell<FxHashMap<(ast::NodeId, &'tcx Substs<'tcx>), ValueRef>>,

    /// Cache of external const values
    extern_const_values: RefCell<DefIdMap<ValueRef>>,

    /// Mapping from static definitions to their DefId's.
    statics: RefCell<FxHashMap<ValueRef, DefId>>,

    impl_method_cache: RefCell<FxHashMap<(DefId, ast::Name), DefId>>,

    /// Cache of closure wrappers for bare fn's.
    closure_bare_wrapper_cache: RefCell<FxHashMap<ValueRef, ValueRef>>,

    /// List of globals for static variables which need to be passed to the
    /// LLVM function ReplaceAllUsesWith (RAUW) when translation is complete.
    /// (We have to make sure we don't invalidate any ValueRefs referring
    /// to constants.)
    statics_to_rauw: RefCell<Vec<(ValueRef, ValueRef)>>,

    lltypes: RefCell<FxHashMap<Ty<'tcx>, Type>>,
    llsizingtypes: RefCell<FxHashMap<Ty<'tcx>, Type>>,
    type_hashcodes: RefCell<FxHashMap<Ty<'tcx>, String>>,
    int_type: Type,
    opaque_vec_type: Type,
    str_slice_type: Type,

    /// Holds the LLVM values for closure IDs.
    closure_vals: RefCell<FxHashMap<Instance<'tcx>, ValueRef>>,

    dbg_cx: Option<debuginfo::CrateDebugContext<'tcx>>,

    eh_personality: Cell<Option<ValueRef>>,
    eh_unwind_resume: Cell<Option<ValueRef>>,
    rust_try_fn: Cell<Option<ValueRef>>,

    intrinsics: RefCell<FxHashMap<&'static str, ValueRef>>,

    /// Depth of the current type-of computation - used to bail out
    type_of_depth: Cell<usize>,

    symbol_map: Rc<SymbolMap<'tcx>>,

    /// A counter that is used for generating local symbol names
    local_gen_sym_counter: Cell<usize>,
}

// Implement DepTrackingMapConfig for `trait_cache`
pub struct TraitSelectionCache<'tcx> {
    data: PhantomData<&'tcx ()>
}

impl<'tcx> DepTrackingMapConfig for TraitSelectionCache<'tcx> {
    type Key = ty::PolyTraitRef<'tcx>;
    type Value = traits::Vtable<'tcx, ()>;
    fn to_dep_node(key: &ty::PolyTraitRef<'tcx>) -> DepNode<DefId> {
        key.to_poly_trait_predicate().dep_node()
    }
}

// # Global Cache

pub struct ProjectionCache<'gcx> {
    data: PhantomData<&'gcx ()>
}

impl<'gcx> DepTrackingMapConfig for ProjectionCache<'gcx> {
    type Key = Ty<'gcx>;
    type Value = Ty<'gcx>;
    fn to_dep_node(key: &Self::Key) -> DepNode<DefId> {
        // Ideally, we'd just put `key` into the dep-node, but we
        // can't put full types in there. So just collect up all the
        // def-ids of structs/enums as well as any traits that we
        // project out of. It doesn't matter so much what we do here,
        // except that if we are too coarse, we'll create overly
        // coarse edges between impls and the trans. For example, if
        // we just used the def-id of things we are projecting out of,
        // then the key for `<Foo as SomeTrait>::T` and `<Bar as
        // SomeTrait>::T` would both share a dep-node
        // (`TraitSelect(SomeTrait)`), and hence the impls for both
        // `Foo` and `Bar` would be considered inputs. So a change to
        // `Bar` would affect things that just normalized `Foo`.
        // Anyway, this heuristic is not ideal, but better than
        // nothing.
        let def_ids: Vec<DefId> =
            key.walk()
               .filter_map(|t| match t.sty {
                   ty::TyAdt(adt_def, _) => Some(adt_def.did),
                   ty::TyProjection(ref proj) => Some(proj.trait_ref.def_id),
                   _ => None,
               })
               .collect();
        DepNode::TraitSelect(def_ids)
    }
}

/// This list owns a number of LocalCrateContexts and binds them to their common
/// SharedCrateContext. This type just exists as a convenience, something to
/// pass around all LocalCrateContexts with and get an iterator over them.
pub struct CrateContextList<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    local_ccxs: Vec<LocalCrateContext<'tcx>>,
}

impl<'a, 'tcx: 'a> CrateContextList<'a, 'tcx> {
    pub fn new(shared_ccx: &'a SharedCrateContext<'a, 'tcx>,
               codegen_units: Vec<CodegenUnit<'tcx>>,
               previous_work_products: Vec<Option<WorkProduct>>,
               symbol_map: Rc<SymbolMap<'tcx>>)
               -> CrateContextList<'a, 'tcx> {
        CrateContextList {
            shared: shared_ccx,
            local_ccxs: codegen_units.into_iter().zip(previous_work_products).map(|(cgu, wp)| {
                LocalCrateContext::new(shared_ccx, cgu, wp, symbol_map.clone())
            }).collect()
        }
    }

    /// Iterate over all crate contexts, whether or not they need
    /// translation.  That is, whether or not a `.o` file is available
    /// for re-use from a previous incr. comp.).
    pub fn iter_all<'b>(&'b self) -> CrateContextIterator<'b, 'tcx> {
        CrateContextIterator {
            shared: self.shared,
            index: 0,
            local_ccxs: &self.local_ccxs[..],
            filter_to_previous_work_product_unavail: false,
        }
    }

    /// Iterator over all CCX that need translation (cannot reuse results from
    /// previous incr. comp.).
    pub fn iter_need_trans<'b>(&'b self) -> CrateContextIterator<'b, 'tcx> {
        CrateContextIterator {
            shared: self.shared,
            index: 0,
            local_ccxs: &self.local_ccxs[..],
            filter_to_previous_work_product_unavail: true,
        }
    }

    pub fn shared(&self) -> &'a SharedCrateContext<'a, 'tcx> {
        self.shared
    }
}

/// A CrateContext value binds together one LocalCrateContext with the
/// SharedCrateContext. It exists as a convenience wrapper, so we don't have to
/// pass around (SharedCrateContext, LocalCrateContext) tuples all over trans.
pub struct CrateContext<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    local_ccxs: &'a [LocalCrateContext<'tcx>],
    /// The index of `local` in `local_ccxs`.  This is used in
    /// `maybe_iter(true)` to identify the original `LocalCrateContext`.
    index: usize,
}

pub struct CrateContextIterator<'a, 'tcx: 'a> {
    shared: &'a SharedCrateContext<'a, 'tcx>,
    local_ccxs: &'a [LocalCrateContext<'tcx>],
    index: usize,

    /// if true, only return results where `previous_work_product` is none
    filter_to_previous_work_product_unavail: bool,
}

impl<'a, 'tcx> Iterator for CrateContextIterator<'a,'tcx> {
    type Item = CrateContext<'a, 'tcx>;

    fn next(&mut self) -> Option<CrateContext<'a, 'tcx>> {
        loop {
            if self.index >= self.local_ccxs.len() {
                return None;
            }

            let index = self.index;
            self.index += 1;

            let ccx = CrateContext {
                shared: self.shared,
                index: index,
                local_ccxs: self.local_ccxs,
            };

            if
                self.filter_to_previous_work_product_unavail &&
                ccx.previous_work_product().is_some()
            {
                continue;
            }

            return Some(ccx);
        }
    }
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
                             sess.opts
                                 .cg
                                 .code_model));
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

unsafe fn create_context_and_module(sess: &Session, mod_name: &str) -> (ContextRef, ModuleRef) {
    let llcx = llvm::LLVMContextCreate();
    let mod_name = CString::new(mod_name).unwrap();
    let llmod = llvm::LLVMModuleCreateWithNameInContext(mod_name.as_ptr(), llcx);

    // Ensure the data-layout values hardcoded remain the defaults.
    if sess.target.target.options.is_builtin {
        let tm = ::back::write::create_target_machine(sess);
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

impl<'b, 'tcx> SharedCrateContext<'b, 'tcx> {
    pub fn new(tcx: TyCtxt<'b, 'tcx, 'tcx>,
               export_map: ExportMap,
               link_meta: LinkMeta,
               exported_symbols: NodeSet,
               check_overflow: bool)
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

        SharedCrateContext {
            metadata_llmod: metadata_llmod,
            metadata_llcx: metadata_llcx,
            export_map: export_map,
            exported_symbols: exported_symbols,
            link_meta: link_meta,
            empty_param_env: tcx.empty_parameter_environment(),
            tcx: tcx,
            stats: Stats {
                n_glues_created: Cell::new(0),
                n_null_glues: Cell::new(0),
                n_real_glues: Cell::new(0),
                n_fns: Cell::new(0),
                n_inlines: Cell::new(0),
                n_closures: Cell::new(0),
                n_llvm_insns: Cell::new(0),
                llvm_insns: RefCell::new(FxHashMap()),
                fn_stats: RefCell::new(Vec::new()),
            },
            check_overflow: check_overflow,
            use_dll_storage_attrs: use_dll_storage_attrs,
            translation_items: RefCell::new(FxHashSet()),
            trait_cache: RefCell::new(DepTrackingMap::new(tcx.dep_graph.clone())),
            project_cache: RefCell::new(DepTrackingMap::new(tcx.dep_graph.clone())),
        }
    }

    pub fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_needs_drop_given_env(ty, &self.empty_param_env)
    }

    pub fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx, &self.empty_param_env, DUMMY_SP)
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

    pub fn exported_symbols<'a>(&'a self) -> &'a NodeSet {
        &self.exported_symbols
    }

    pub fn trait_cache(&self) -> &RefCell<DepTrackingMap<TraitSelectionCache<'tcx>>> {
        &self.trait_cache
    }

    pub fn project_cache(&self) -> &RefCell<DepTrackingMap<ProjectionCache<'tcx>>> {
        &self.project_cache
    }

    pub fn link_meta<'a>(&'a self) -> &'a LinkMeta {
        &self.link_meta
    }

    pub fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    pub fn dep_graph<'a>(&'a self) -> &'a DepGraph {
        &self.tcx.dep_graph
    }

    pub fn stats<'a>(&'a self) -> &'a Stats {
        &self.stats
    }

    pub fn use_dll_storage_attrs(&self) -> bool {
        self.use_dll_storage_attrs
    }

    pub fn translation_items(&self) -> &RefCell<FxHashSet<TransItem<'tcx>>> {
        &self.translation_items
    }

    /// Given the def-id of some item that has no type parameters, make
    /// a suitable "empty substs" for it.
    pub fn empty_substs_for_def_id(&self, item_def_id: DefId) -> &'tcx Substs<'tcx> {
        Substs::for_item(self.tcx(), item_def_id,
                         |_, _| self.tcx().mk_region(ty::ReErased),
                         |_, _| {
            bug!("empty_substs_for_def_id: {:?} has type parameters", item_def_id)
        })
    }

    pub fn metadata_symbol_name(&self) -> String {
        format!("rust_metadata_{}_{}",
                self.link_meta().crate_name,
                self.link_meta().crate_hash)
    }
}

impl<'tcx> LocalCrateContext<'tcx> {
    fn new<'a>(shared: &SharedCrateContext<'a, 'tcx>,
               codegen_unit: CodegenUnit<'tcx>,
               previous_work_product: Option<WorkProduct>,
               symbol_map: Rc<SymbolMap<'tcx>>)
           -> LocalCrateContext<'tcx> {
        unsafe {
            // Append ".rs" to LLVM module identifier.
            //
            // LLVM code generator emits a ".file filename" directive
            // for ELF backends. Value of the "filename" is set as the
            // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
            // crashes if the module identifier is same as other symbols
            // such as a function name in the module.
            // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
            let llmod_id = format!("{}.rs", codegen_unit.name());

            let (llcx, llmod) = create_context_and_module(&shared.tcx.sess,
                                                          &llmod_id[..]);

            let dbg_cx = if shared.tcx.sess.opts.debuginfo != NoDebugInfo {
                let dctx = debuginfo::CrateDebugContext::new(llmod);
                debuginfo::metadata::compile_unit_metadata(shared, &dctx, shared.tcx.sess);
                Some(dctx)
            } else {
                None
            };

            let local_ccx = LocalCrateContext {
                llmod: llmod,
                llcx: llcx,
                previous_work_product: previous_work_product,
                codegen_unit: codegen_unit,
                needs_unwind_cleanup_cache: RefCell::new(FxHashMap()),
                fn_pointer_shims: RefCell::new(FxHashMap()),
                drop_glues: RefCell::new(FxHashMap()),
                instances: RefCell::new(FxHashMap()),
                vtables: RefCell::new(FxHashMap()),
                const_cstr_cache: RefCell::new(FxHashMap()),
                const_unsized: RefCell::new(FxHashMap()),
                const_globals: RefCell::new(FxHashMap()),
                const_values: RefCell::new(FxHashMap()),
                extern_const_values: RefCell::new(DefIdMap()),
                statics: RefCell::new(FxHashMap()),
                impl_method_cache: RefCell::new(FxHashMap()),
                closure_bare_wrapper_cache: RefCell::new(FxHashMap()),
                statics_to_rauw: RefCell::new(Vec::new()),
                lltypes: RefCell::new(FxHashMap()),
                llsizingtypes: RefCell::new(FxHashMap()),
                type_hashcodes: RefCell::new(FxHashMap()),
                int_type: Type::from_ref(ptr::null_mut()),
                opaque_vec_type: Type::from_ref(ptr::null_mut()),
                str_slice_type: Type::from_ref(ptr::null_mut()),
                closure_vals: RefCell::new(FxHashMap()),
                dbg_cx: dbg_cx,
                eh_personality: Cell::new(None),
                eh_unwind_resume: Cell::new(None),
                rust_try_fn: Cell::new(None),
                intrinsics: RefCell::new(FxHashMap()),
                type_of_depth: Cell::new(0),
                symbol_map: symbol_map,
                local_gen_sym_counter: Cell::new(0),
            };

            let (int_type, opaque_vec_type, str_slice_ty, mut local_ccx) = {
                // Do a little dance to create a dummy CrateContext, so we can
                // create some things in the LLVM module of this codegen unit
                let mut local_ccxs = vec![local_ccx];
                let (int_type, opaque_vec_type, str_slice_ty) = {
                    let dummy_ccx = LocalCrateContext::dummy_ccx(shared,
                                                                 local_ccxs.as_mut_slice());
                    let mut str_slice_ty = Type::named_struct(&dummy_ccx, "str_slice");
                    str_slice_ty.set_struct_body(&[Type::i8p(&dummy_ccx),
                                                   Type::int(&dummy_ccx)],
                                                 false);
                    (Type::int(&dummy_ccx), Type::opaque_vec(&dummy_ccx), str_slice_ty)
                };
                (int_type, opaque_vec_type, str_slice_ty, local_ccxs.pop().unwrap())
            };

            local_ccx.int_type = int_type;
            local_ccx.opaque_vec_type = opaque_vec_type;
            local_ccx.str_slice_type = str_slice_ty;

            local_ccx
        }
    }

    /// Create a dummy `CrateContext` from `self` and  the provided
    /// `SharedCrateContext`.  This is somewhat dangerous because `self` may
    /// not be fully initialized.
    ///
    /// This is used in the `LocalCrateContext` constructor to allow calling
    /// functions that expect a complete `CrateContext`, even before the local
    /// portion is fully initialized and attached to the `SharedCrateContext`.
    fn dummy_ccx<'a>(shared: &'a SharedCrateContext<'a, 'tcx>,
                     local_ccxs: &'a [LocalCrateContext<'tcx>])
                     -> CrateContext<'a, 'tcx> {
        assert!(local_ccxs.len() == 1);
        CrateContext {
            shared: shared,
            index: 0,
            local_ccxs: local_ccxs
        }
    }
}

impl<'b, 'tcx> CrateContext<'b, 'tcx> {
    pub fn shared(&self) -> &'b SharedCrateContext<'b, 'tcx> {
        self.shared
    }

    fn local(&self) -> &'b LocalCrateContext<'tcx> {
        &self.local_ccxs[self.index]
    }

    pub fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.shared.tcx
    }

    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.shared.tcx.sess
    }

    pub fn get_intrinsic(&self, key: &str) -> ValueRef {
        if let Some(v) = self.intrinsics().borrow().get(key).cloned() {
            return v;
        }
        match declare_intrinsic(self, key) {
            Some(v) => return v,
            None => bug!("unknown intrinsic '{}'", key)
        }
    }

    pub fn llmod(&self) -> ModuleRef {
        self.local().llmod
    }

    pub fn llcx(&self) -> ContextRef {
        self.local().llcx
    }

    pub fn previous_work_product(&self) -> Option<&WorkProduct> {
        self.local().previous_work_product.as_ref()
    }

    pub fn codegen_unit(&self) -> &CodegenUnit<'tcx> {
        &self.local().codegen_unit
    }

    pub fn td(&self) -> llvm::TargetDataRef {
        unsafe { llvm::LLVMRustGetModuleDataLayout(self.llmod()) }
    }

    pub fn export_map<'a>(&'a self) -> &'a ExportMap {
        &self.shared.export_map
    }

    pub fn exported_symbols<'a>(&'a self) -> &'a NodeSet {
        &self.shared.exported_symbols
    }

    pub fn link_meta<'a>(&'a self) -> &'a LinkMeta {
        &self.shared.link_meta
    }

    pub fn needs_unwind_cleanup_cache(&self) -> &RefCell<FxHashMap<Ty<'tcx>, bool>> {
        &self.local().needs_unwind_cleanup_cache
    }

    pub fn fn_pointer_shims(&self) -> &RefCell<FxHashMap<Ty<'tcx>, ValueRef>> {
        &self.local().fn_pointer_shims
    }

    pub fn drop_glues<'a>(&'a self)
                          -> &'a RefCell<FxHashMap<DropGlueKind<'tcx>, (ValueRef, FnType)>> {
        &self.local().drop_glues
    }

    pub fn instances<'a>(&'a self) -> &'a RefCell<FxHashMap<Instance<'tcx>, ValueRef>> {
        &self.local().instances
    }

    pub fn vtables<'a>(&'a self)
        -> &'a RefCell<FxHashMap<(ty::Ty<'tcx>,
                                  Option<ty::PolyExistentialTraitRef<'tcx>>), ValueRef>> {
        &self.local().vtables
    }

    pub fn const_cstr_cache<'a>(&'a self) -> &'a RefCell<FxHashMap<InternedString, ValueRef>> {
        &self.local().const_cstr_cache
    }

    pub fn const_unsized<'a>(&'a self) -> &'a RefCell<FxHashMap<ValueRef, ValueRef>> {
        &self.local().const_unsized
    }

    pub fn const_globals<'a>(&'a self) -> &'a RefCell<FxHashMap<ValueRef, ValueRef>> {
        &self.local().const_globals
    }

    pub fn const_values<'a>(&'a self) -> &'a RefCell<FxHashMap<(ast::NodeId, &'tcx Substs<'tcx>),
                                                               ValueRef>> {
        &self.local().const_values
    }

    pub fn extern_const_values<'a>(&'a self) -> &'a RefCell<DefIdMap<ValueRef>> {
        &self.local().extern_const_values
    }

    pub fn statics<'a>(&'a self) -> &'a RefCell<FxHashMap<ValueRef, DefId>> {
        &self.local().statics
    }

    pub fn impl_method_cache<'a>(&'a self)
            -> &'a RefCell<FxHashMap<(DefId, ast::Name), DefId>> {
        &self.local().impl_method_cache
    }

    pub fn closure_bare_wrapper_cache<'a>(&'a self) -> &'a RefCell<FxHashMap<ValueRef, ValueRef>> {
        &self.local().closure_bare_wrapper_cache
    }

    pub fn statics_to_rauw<'a>(&'a self) -> &'a RefCell<Vec<(ValueRef, ValueRef)>> {
        &self.local().statics_to_rauw
    }

    pub fn lltypes<'a>(&'a self) -> &'a RefCell<FxHashMap<Ty<'tcx>, Type>> {
        &self.local().lltypes
    }

    pub fn llsizingtypes<'a>(&'a self) -> &'a RefCell<FxHashMap<Ty<'tcx>, Type>> {
        &self.local().llsizingtypes
    }

    pub fn type_hashcodes<'a>(&'a self) -> &'a RefCell<FxHashMap<Ty<'tcx>, String>> {
        &self.local().type_hashcodes
    }

    pub fn stats<'a>(&'a self) -> &'a Stats {
        &self.shared.stats
    }

    pub fn int_type(&self) -> Type {
        self.local().int_type
    }

    pub fn opaque_vec_type(&self) -> Type {
        self.local().opaque_vec_type
    }

    pub fn str_slice_type(&self) -> Type {
        self.local().str_slice_type
    }

    pub fn closure_vals<'a>(&'a self) -> &'a RefCell<FxHashMap<Instance<'tcx>, ValueRef>> {
        &self.local().closure_vals
    }

    pub fn dbg_cx<'a>(&'a self) -> &'a Option<debuginfo::CrateDebugContext<'tcx>> {
        &self.local().dbg_cx
    }

    pub fn rust_try_fn<'a>(&'a self) -> &'a Cell<Option<ValueRef>> {
        &self.local().rust_try_fn
    }

    fn intrinsics<'a>(&'a self) -> &'a RefCell<FxHashMap<&'static str, ValueRef>> {
        &self.local().intrinsics
    }

    pub fn obj_size_bound(&self) -> u64 {
        self.tcx().data_layout.obj_size_bound()
    }

    pub fn report_overbig_object(&self, obj: Ty<'tcx>) -> ! {
        self.sess().fatal(
            &format!("the type `{:?}` is too big for the current architecture",
                    obj))
    }

    pub fn enter_type_of(&self, ty: Ty<'tcx>) -> TypeOfDepthLock<'b, 'tcx> {
        let current_depth = self.local().type_of_depth.get();
        debug!("enter_type_of({:?}) at depth {:?}", ty, current_depth);
        if current_depth > self.sess().recursion_limit.get() {
            self.sess().fatal(
                &format!("overflow representing the type `{}`", ty))
        }
        self.local().type_of_depth.set(current_depth + 1);
        TypeOfDepthLock(self.local())
    }

    pub fn layout_of(&self, ty: Ty<'tcx>) -> &'tcx ty::layout::Layout {
        self.tcx().infer_ctxt((), traits::Reveal::All).enter(|infcx| {
            ty.layout(&infcx).unwrap_or_else(|e| {
                match e {
                    ty::layout::LayoutError::SizeOverflow(_) =>
                        self.sess().fatal(&e.to_string()),
                    _ => bug!("failed to get layout for `{}`: {}", ty, e)
                }
            })
        })
    }

    pub fn check_overflow(&self) -> bool {
        self.shared.check_overflow
    }

    pub fn use_dll_storage_attrs(&self) -> bool {
        self.shared.use_dll_storage_attrs()
    }

    pub fn symbol_map(&self) -> &SymbolMap<'tcx> {
        &*self.local().symbol_map
    }

    pub fn translation_items(&self) -> &RefCell<FxHashSet<TransItem<'tcx>>> {
        &self.shared.translation_items
    }

    /// Given the def-id of some item that has no type parameters, make
    /// a suitable "empty substs" for it.
    pub fn empty_substs_for_def_id(&self, item_def_id: DefId) -> &'tcx Substs<'tcx> {
        self.shared().empty_substs_for_def_id(item_def_id)
    }

    /// Generate a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local().local_gen_sym_counter.get();
        self.local().local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push_str(".");
        base_n::push_str(idx as u64, base_n::ALPHANUMERIC_ONLY, &mut name);
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
        if let Some(llpersonality) = self.local().eh_personality.get() {
            return llpersonality
        }
        let tcx = self.tcx();
        let llfn = match tcx.lang_items.eh_personality() {
            Some(def_id) if !base::wants_msvc_seh(self.sess()) => {
                Callee::def(self, def_id, tcx.intern_substs(&[])).reify(self)
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
        self.local().eh_personality.set(Some(llfn));
        llfn
    }

    // Returns a ValueRef of the "eh_unwind_resume" lang item if one is defined,
    // otherwise declares it as an external function.
    pub fn eh_unwind_resume(&self) -> ValueRef {
        use attributes;
        let unwresume = &self.local().eh_unwind_resume;
        if let Some(llfn) = unwresume.get() {
            return llfn;
        }

        let tcx = self.tcx();
        assert!(self.sess().target.target.options.custom_unwind_resume);
        if let Some(def_id) = tcx.lang_items.eh_unwind_resume() {
            let llfn = Callee::def(self, def_id, tcx.intern_substs(&[])).reify(self);
            unwresume.set(Some(llfn));
            return llfn;
        }

        let ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: hir::Unsafety::Unsafe,
            abi: Abi::C,
            sig: ty::Binder(tcx.mk_fn_sig(
                iter::once(tcx.mk_mut_ptr(tcx.types.u8)),
                tcx.types.never,
                false
            )),
        }));

        let llfn = declare::declare_fn(self, "rust_eh_unwind_resume", ty);
        attributes::unwind(llfn, true);
        unwresume.set(Some(llfn));
        llfn
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
                let f = declare::declare_cfn(ccx, $name, Type::func(&[], &$ret));
                llvm::SetUnnamedAddr(f, false);
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, fn(...) -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(ccx, $name, Type::variadic_func(&[], &$ret));
                llvm::SetUnnamedAddr(f, false);
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr, fn($($arg:expr),*) -> $ret:expr) => (
            if key == $name {
                let f = declare::declare_cfn(ccx, $name, Type::func(&[$($arg),*], &$ret));
                llvm::SetUnnamedAddr(f, false);
                ccx.intrinsics().borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
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
    let t_i128 = Type::i128(ccx);
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

    if ccx.sess().opts.debuginfo != NoDebugInfo {
        ifn!("llvm.dbg.declare", fn(Type::metadata(ccx), Type::metadata(ccx)) -> void);
        ifn!("llvm.dbg.value", fn(Type::metadata(ccx), t_i64, Type::metadata(ccx)) -> void);
    }
    return None;
}
