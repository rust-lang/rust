//! Lowers the AST to the HIR.
//!
//! Since the AST and HIR are fairly similar, this is mostly a simple procedure,
//! much like a fold. Where lowering involves a bit more work things get more
//! interesting and there are some invariants you should know about. These mostly
//! concern spans and IDs.
//!
//! Spans are assigned to AST nodes during parsing and then are modified during
//! expansion to indicate the origin of a node and the process it went through
//! being expanded. IDs are assigned to AST nodes just before lowering.
//!
//! For the simpler lowering steps, IDs and spans should be preserved. Unlike
//! expansion we do not preserve the process of lowering in the spans, so spans
//! should not be modified here. When creating a new node (as opposed to
//! "folding" an existing one), create a new ID using `next_id()`.
//!
//! You must ensure that IDs are unique. That means that you should only use the
//! ID from an AST node in a single HIR node (you can assume that AST node-IDs
//! are unique). Every new node must have a unique ID. Avoid cloning HIR nodes.
//! If you do, you must then set the new node's ID to a fresh one.
//!
//! Spans are used for error messages and for tools to map semantics back to
//! source code. It is therefore not as important with spans as IDs to be strict
//! about use (you can't break the compiler by screwing up a span). Obviously, a
//! HIR node can only have a single span. But multiple nodes can have the same
//! span and spans don't need to be kept in order, etc. Where code is preserved
//! by lowering, it should have the same span as in the AST. Where HIR nodes are
//! new it is probably best to give a span for the whole AST node being lowered.
//! All nodes should have real spans; don't use dummy spans. Tools are likely to
//! get confused if the spans from leaf AST nodes occur in multiple places
//! in the HIR, especially for multiple identifiers.

#![cfg_attr(not(bootstrap), allow(internal_features))]
#![cfg_attr(not(bootstrap), feature(rustdoc_internals))]
#![cfg_attr(not(bootstrap), doc(rust_logo))]
#![feature(box_patterns)]
#![feature(let_chains)]
#![feature(never_type)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;

use crate::errors::{AssocTyParentheses, AssocTyParenthesesSub, MisplacedImplTrait};

use rustc_ast::ptr::P;
use rustc_ast::visit;
use rustc_ast::{self as ast, *};
use rustc_ast_pretty::pprust;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{
    DiagnosticArgFromDisplay, DiagnosticMessage, Handler, StashKey, SubdiagnosticMessage,
};
use rustc_fluent_macro::fluent_messages;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, LifetimeRes, Namespace, PartialRes, PerNS, Res};
use rustc_hir::def_id::{LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::DefPathData;
use rustc_hir::{ConstArg, GenericArg, ItemLocalId, ParamName, TraitCandidate};
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_middle::{
    span_bug,
    ty::{ResolverAstLowering, TyCtxt},
};
use rustc_session::parse::{add_feature_diagnostics, feature_err};
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::DesugaringKind;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use thin_vec::ThinVec;

macro_rules! arena_vec {
    ($this:expr; $($x:expr),*) => (
        $this.arena.alloc_from_iter([$($x),*])
    );
}

mod asm;
mod block;
mod errors;
mod expr;
mod format;
mod index;
mod item;
mod lifetime_collector;
mod pat;
mod path;

fluent_messages! { "../messages.ftl" }

struct LoweringContext<'a, 'hir> {
    tcx: TyCtxt<'hir>,
    resolver: &'a mut ResolverAstLowering,

    /// Used to allocate HIR nodes.
    arena: &'hir hir::Arena<'hir>,

    /// Bodies inside the owner being lowered.
    bodies: Vec<(hir::ItemLocalId, &'hir hir::Body<'hir>)>,
    /// Attributes inside the owner being lowered.
    attrs: SortedMap<hir::ItemLocalId, &'hir [Attribute]>,
    /// Collect items that were created by lowering the current owner.
    children: Vec<(LocalDefId, hir::MaybeOwner<&'hir hir::OwnerInfo<'hir>>)>,

    coroutine_kind: Option<hir::CoroutineKind>,

    /// When inside an `async` context, this is the `HirId` of the
    /// `task_context` local bound to the resume argument of the coroutine.
    task_context: Option<hir::HirId>,

    /// Used to get the current `fn`'s def span to point to when using `await`
    /// outside of an `async fn`.
    current_item: Option<Span>,

    catch_scope: Option<NodeId>,
    loop_scope: Option<NodeId>,
    is_in_loop_condition: bool,
    is_in_trait_impl: bool,
    is_in_dyn_type: bool,

    current_hir_id_owner: hir::OwnerId,
    item_local_id_counter: hir::ItemLocalId,
    trait_map: FxHashMap<ItemLocalId, Box<[TraitCandidate]>>,

    impl_trait_defs: Vec<hir::GenericParam<'hir>>,
    impl_trait_bounds: Vec<hir::WherePredicate<'hir>>,

    /// NodeIds that are lowered inside the current HIR owner.
    node_id_to_local_id: FxHashMap<NodeId, hir::ItemLocalId>,

    allow_try_trait: Option<Lrc<[Symbol]>>,
    allow_gen_future: Option<Lrc<[Symbol]>>,

    /// Mapping from generics `def_id`s to TAIT generics `def_id`s.
    /// For each captured lifetime (e.g., 'a), we create a new lifetime parameter that is a generic
    /// defined on the TAIT, so we have type Foo<'a1> = ... and we establish a mapping in this
    /// field from the original parameter 'a to the new parameter 'a1.
    generics_def_id_map: Vec<FxHashMap<LocalDefId, LocalDefId>>,

    host_param_id: Option<LocalDefId>,
}

trait ResolverAstLoweringExt {
    fn legacy_const_generic_args(&self, expr: &Expr) -> Option<Vec<usize>>;
    fn get_partial_res(&self, id: NodeId) -> Option<PartialRes>;
    fn get_import_res(&self, id: NodeId) -> PerNS<Option<Res<NodeId>>>;
    fn get_label_res(&self, id: NodeId) -> Option<NodeId>;
    fn get_lifetime_res(&self, id: NodeId) -> Option<LifetimeRes>;
    fn take_extra_lifetime_params(&mut self, id: NodeId) -> Vec<(Ident, NodeId, LifetimeRes)>;
    fn remap_extra_lifetime_params(&mut self, from: NodeId, to: NodeId);
    fn decl_macro_kind(&self, def_id: LocalDefId) -> MacroKind;
}

impl ResolverAstLoweringExt for ResolverAstLowering {
    fn legacy_const_generic_args(&self, expr: &Expr) -> Option<Vec<usize>> {
        if let ExprKind::Path(None, path) = &expr.kind {
            // Don't perform legacy const generics rewriting if the path already
            // has generic arguments.
            if path.segments.last().unwrap().args.is_some() {
                return None;
            }

            if let Res::Def(DefKind::Fn, def_id) = self.partial_res_map.get(&expr.id)?.full_res()? {
                // We only support cross-crate argument rewriting. Uses
                // within the same crate should be updated to use the new
                // const generics style.
                if def_id.is_local() {
                    return None;
                }

                if let Some(v) = self.legacy_const_generic_args.get(&def_id) {
                    return v.clone();
                }
            }
        }

        None
    }

    /// Obtains resolution for a `NodeId` with a single resolution.
    fn get_partial_res(&self, id: NodeId) -> Option<PartialRes> {
        self.partial_res_map.get(&id).copied()
    }

    /// Obtains per-namespace resolutions for `use` statement with the given `NodeId`.
    fn get_import_res(&self, id: NodeId) -> PerNS<Option<Res<NodeId>>> {
        self.import_res_map.get(&id).copied().unwrap_or_default()
    }

    /// Obtains resolution for a label with the given `NodeId`.
    fn get_label_res(&self, id: NodeId) -> Option<NodeId> {
        self.label_res_map.get(&id).copied()
    }

    /// Obtains resolution for a lifetime with the given `NodeId`.
    fn get_lifetime_res(&self, id: NodeId) -> Option<LifetimeRes> {
        self.lifetimes_res_map.get(&id).copied()
    }

    /// Obtain the list of lifetimes parameters to add to an item.
    ///
    /// Extra lifetime parameters should only be added in places that can appear
    /// as a `binder` in `LifetimeRes`.
    ///
    /// The extra lifetimes that appear from the parenthesized `Fn`-trait desugaring
    /// should appear at the enclosing `PolyTraitRef`.
    fn take_extra_lifetime_params(&mut self, id: NodeId) -> Vec<(Ident, NodeId, LifetimeRes)> {
        self.extra_lifetime_params_map.remove(&id).unwrap_or_default()
    }

    fn remap_extra_lifetime_params(&mut self, from: NodeId, to: NodeId) {
        let lifetimes = self.extra_lifetime_params_map.remove(&from).unwrap_or_default();
        self.extra_lifetime_params_map.insert(to, lifetimes);
    }

    fn decl_macro_kind(&self, def_id: LocalDefId) -> MacroKind {
        self.builtin_macro_kinds.get(&def_id).copied().unwrap_or(MacroKind::Bang)
    }
}

/// Context of `impl Trait` in code, which determines whether it is allowed in an HIR subtree,
/// and if so, what meaning it has.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitContext {
    /// Treat `impl Trait` as shorthand for a new universal generic parameter.
    /// Example: `fn foo(x: impl Debug)`, where `impl Debug` is conceptually
    /// equivalent to a fresh universal parameter like `fn foo<T: Debug>(x: T)`.
    ///
    /// Newly generated parameters should be inserted into the given `Vec`.
    Universal,

    /// Treat `impl Trait` as shorthand for a new opaque type.
    /// Example: `fn foo() -> impl Debug`, where `impl Debug` is conceptually
    /// equivalent to a new opaque type like `type T = impl Debug; fn foo() -> T`.
    ///
    ReturnPositionOpaqueTy {
        /// Origin: Either OpaqueTyOrigin::FnReturn or OpaqueTyOrigin::AsyncFn,
        origin: hir::OpaqueTyOrigin,
        fn_kind: FnDeclKind,
    },
    /// Impl trait in type aliases.
    TypeAliasesOpaqueTy { in_assoc_ty: bool },
    /// `impl Trait` is unstably accepted in this position.
    FeatureGated(ImplTraitPosition, Symbol),
    /// `impl Trait` is not accepted in this position.
    Disallowed(ImplTraitPosition),
}

/// Position in which `impl Trait` is disallowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitPosition {
    Path,
    Variable,
    Trait,
    AsyncBlock,
    Bound,
    Generic,
    ExternFnParam,
    ClosureParam,
    PointerParam,
    FnTraitParam,
    TraitParam,
    ImplParam,
    ExternFnReturn,
    ClosureReturn,
    PointerReturn,
    FnTraitReturn,
    GenericDefault,
    ConstTy,
    StaticTy,
    AssocTy,
    FieldTy,
    Cast,
    ImplSelf,
    OffsetOf,
}

impl std::fmt::Display for ImplTraitPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ImplTraitPosition::Path => "paths",
            ImplTraitPosition::Variable => "variable bindings",
            ImplTraitPosition::Trait => "traits",
            ImplTraitPosition::AsyncBlock => "async blocks",
            ImplTraitPosition::Bound => "bounds",
            ImplTraitPosition::Generic => "generics",
            ImplTraitPosition::ExternFnParam => "`extern fn` params",
            ImplTraitPosition::ClosureParam => "closure params",
            ImplTraitPosition::PointerParam => "`fn` pointer params",
            ImplTraitPosition::FnTraitParam => "`Fn` trait params",
            ImplTraitPosition::TraitParam => "trait method params",
            ImplTraitPosition::ImplParam => "`impl` method params",
            ImplTraitPosition::ExternFnReturn => "`extern fn` return types",
            ImplTraitPosition::ClosureReturn => "closure return types",
            ImplTraitPosition::PointerReturn => "`fn` pointer return types",
            ImplTraitPosition::FnTraitReturn => "`Fn` trait return types",
            ImplTraitPosition::GenericDefault => "generic parameter defaults",
            ImplTraitPosition::ConstTy => "const types",
            ImplTraitPosition::StaticTy => "static types",
            ImplTraitPosition::AssocTy => "associated types",
            ImplTraitPosition::FieldTy => "field types",
            ImplTraitPosition::Cast => "cast types",
            ImplTraitPosition::ImplSelf => "impl headers",
            ImplTraitPosition::OffsetOf => "`offset_of!` params",
        };

        write!(f, "{name}")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FnDeclKind {
    Fn,
    Inherent,
    ExternFn,
    Closure,
    Pointer,
    Trait,
    Impl,
}

impl FnDeclKind {
    fn param_impl_trait_allowed(&self) -> bool {
        matches!(self, FnDeclKind::Fn | FnDeclKind::Inherent | FnDeclKind::Impl | FnDeclKind::Trait)
    }

    fn return_impl_trait_allowed(&self) -> bool {
        match self {
            FnDeclKind::Fn | FnDeclKind::Inherent | FnDeclKind::Impl | FnDeclKind::Trait => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone)]
enum AstOwner<'a> {
    NonOwner,
    Crate(&'a ast::Crate),
    Item(&'a ast::Item),
    AssocItem(&'a ast::AssocItem, visit::AssocCtxt),
    ForeignItem(&'a ast::ForeignItem),
}

fn index_crate<'a>(
    node_id_to_def_id: &FxHashMap<NodeId, LocalDefId>,
    krate: &'a Crate,
) -> IndexVec<LocalDefId, AstOwner<'a>> {
    let mut indexer = Indexer { node_id_to_def_id, index: IndexVec::new() };
    *indexer.index.ensure_contains_elem(CRATE_DEF_ID, || AstOwner::NonOwner) =
        AstOwner::Crate(krate);
    visit::walk_crate(&mut indexer, krate);
    return indexer.index;

    struct Indexer<'s, 'a> {
        node_id_to_def_id: &'s FxHashMap<NodeId, LocalDefId>,
        index: IndexVec<LocalDefId, AstOwner<'a>>,
    }

    impl<'a> visit::Visitor<'a> for Indexer<'_, 'a> {
        fn visit_attribute(&mut self, _: &'a Attribute) {
            // We do not want to lower expressions that appear in attributes,
            // as they are not accessible to the rest of the HIR.
        }

        fn visit_item(&mut self, item: &'a ast::Item) {
            let def_id = self.node_id_to_def_id[&item.id];
            *self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner) = AstOwner::Item(item);
            visit::walk_item(self, item)
        }

        fn visit_assoc_item(&mut self, item: &'a ast::AssocItem, ctxt: visit::AssocCtxt) {
            let def_id = self.node_id_to_def_id[&item.id];
            *self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner) =
                AstOwner::AssocItem(item, ctxt);
            visit::walk_assoc_item(self, item, ctxt);
        }

        fn visit_foreign_item(&mut self, item: &'a ast::ForeignItem) {
            let def_id = self.node_id_to_def_id[&item.id];
            *self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner) =
                AstOwner::ForeignItem(item);
            visit::walk_foreign_item(self, item);
        }
    }
}

/// Compute the hash for the HIR of the full crate.
/// This hash will then be part of the crate_hash which is stored in the metadata.
fn compute_hir_hash(
    tcx: TyCtxt<'_>,
    owners: &IndexSlice<LocalDefId, hir::MaybeOwner<&hir::OwnerInfo<'_>>>,
) -> Fingerprint {
    let mut hir_body_nodes: Vec<_> = owners
        .iter_enumerated()
        .filter_map(|(def_id, info)| {
            let info = info.as_owner()?;
            let def_path_hash = tcx.hir().def_path_hash(def_id);
            Some((def_path_hash, info))
        })
        .collect();
    hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

    tcx.with_stable_hashing_context(|mut hcx| {
        let mut stable_hasher = StableHasher::new();
        hir_body_nodes.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    })
}

pub fn lower_to_hir(tcx: TyCtxt<'_>, (): ()) -> hir::Crate<'_> {
    let sess = tcx.sess;
    // Queries that borrow `resolver_for_lowering`.
    tcx.ensure_with_value().output_filenames(());
    tcx.ensure_with_value().early_lint_checks(());
    tcx.ensure_with_value().debugger_visualizers(LOCAL_CRATE);
    let (mut resolver, krate) = tcx.resolver_for_lowering(()).steal();

    let ast_index = index_crate(&resolver.node_id_to_def_id, &krate);
    let mut owners = IndexVec::from_fn_n(
        |_| hir::MaybeOwner::Phantom,
        tcx.definitions_untracked().def_index_count(),
    );

    for def_id in ast_index.indices() {
        item::ItemLowerer {
            tcx,
            resolver: &mut resolver,
            ast_index: &ast_index,
            owners: &mut owners,
        }
        .lower_node(def_id);
    }

    // Drop AST to free memory
    drop(ast_index);
    sess.time("drop_ast", || drop(krate));

    // Discard hygiene data, which isn't required after lowering to HIR.
    if !sess.opts.unstable_opts.keep_hygiene_data {
        rustc_span::hygiene::clear_syntax_context_map();
    }

    // Don't hash unless necessary, because it's expensive.
    let opt_hir_hash =
        if tcx.needs_crate_hash() { Some(compute_hir_hash(tcx, &owners)) } else { None };
    hir::Crate { owners, opt_hir_hash }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum ParamMode {
    /// Any path in a type context.
    Explicit,
    /// Path in a type definition, where the anonymous lifetime `'_` is not allowed.
    ExplicitNamed,
    /// The `module::Type` in `module::Type::method` in an expression.
    Optional,
}

enum ParenthesizedGenericArgs {
    ParenSugar,
    Err,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        data: DefPathData,
        span: Span,
    ) -> LocalDefId {
        debug_assert_ne!(node_id, ast::DUMMY_NODE_ID);
        assert!(
            self.opt_local_def_id(node_id).is_none(),
            "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
            node_id,
            data,
            self.tcx.hir().def_key(self.local_def_id(node_id)),
        );

        let def_id = self.tcx.at(span).create_def(parent, data).def_id();

        debug!("create_def: def_id_to_node_id[{:?}] <-> {:?}", def_id, node_id);
        self.resolver.node_id_to_def_id.insert(node_id, def_id);

        def_id
    }

    fn next_node_id(&mut self) -> NodeId {
        let start = self.resolver.next_node_id;
        let next = start.as_u32().checked_add(1).expect("input too large; ran out of NodeIds");
        self.resolver.next_node_id = ast::NodeId::from_u32(next);
        start
    }

    /// Given the id of some node in the AST, finds the `LocalDefId` associated with it by the name
    /// resolver (if any).
    fn orig_opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.resolver.node_id_to_def_id.get(&node).map(|local_def_id| *local_def_id)
    }

    /// Given the id of some node in the AST, finds the `LocalDefId` associated with it by the name
    /// resolver (if any), after applying any remapping from `get_remapped_def_id`.
    ///
    /// For example, in a function like `fn foo<'a>(x: &'a u32)`,
    /// invoking with the id from the `ast::Lifetime` node found inside
    /// the `&'a u32` type would return the `LocalDefId` of the
    /// `'a` parameter declared on `foo`.
    ///
    /// This function also applies remapping from `get_remapped_def_id`.
    /// These are used when synthesizing opaque types from `-> impl Trait` return types and so forth.
    /// For example, in a function like `fn foo<'a>() -> impl Debug + 'a`,
    /// we would create an opaque type `type FooReturn<'a1> = impl Debug + 'a1`.
    /// When lowering the `Debug + 'a` bounds, we add a remapping to map `'a` to `'a1`.
    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.orig_opt_local_def_id(node).map(|local_def_id| self.get_remapped_def_id(local_def_id))
    }

    fn local_def_id(&self, node: NodeId) -> LocalDefId {
        self.opt_local_def_id(node).unwrap_or_else(|| panic!("no entry for node id: `{node:?}`"))
    }

    /// Get the previously recorded `to` local def id given the `from` local def id, obtained using
    /// `generics_def_id_map` field.
    fn get_remapped_def_id(&self, local_def_id: LocalDefId) -> LocalDefId {
        // `generics_def_id_map` is a stack of mappings. As we go deeper in impl traits nesting we
        // push new mappings, so we first need to get the latest (innermost) mappings, hence `iter().rev()`.
        //
        // Consider:
        //
        // `fn test<'a, 'b>() -> impl Trait<&'a u8, Ty = impl Sized + 'b> {}`
        //
        // We would end with a generics_def_id_map like:
        //
        // `[[fn#'b -> impl_trait#'b], [fn#'b -> impl_sized#'b]]`
        //
        // for the opaque type generated on `impl Sized + 'b`, we want the result to be: impl_sized#'b.
        // So, if we were trying to find first from the start (outermost) would give the wrong result, impl_trait#'b.
        self.generics_def_id_map
            .iter()
            .rev()
            .find_map(|map| map.get(&local_def_id).map(|local_def_id| *local_def_id))
            .unwrap_or(local_def_id)
    }

    /// Freshen the `LoweringContext` and ready it to lower a nested item.
    /// The lowered item is registered into `self.children`.
    ///
    /// This function sets up `HirId` lowering infrastructure,
    /// and stashes the shared mutable state to avoid pollution by the closure.
    #[instrument(level = "debug", skip(self, f))]
    fn with_hir_id_owner(
        &mut self,
        owner: NodeId,
        f: impl FnOnce(&mut Self) -> hir::OwnerNode<'hir>,
    ) {
        let def_id = self.local_def_id(owner);

        let current_attrs = std::mem::take(&mut self.attrs);
        let current_bodies = std::mem::take(&mut self.bodies);
        let current_node_ids = std::mem::take(&mut self.node_id_to_local_id);
        let current_trait_map = std::mem::take(&mut self.trait_map);
        let current_owner =
            std::mem::replace(&mut self.current_hir_id_owner, hir::OwnerId { def_id });
        let current_local_counter =
            std::mem::replace(&mut self.item_local_id_counter, hir::ItemLocalId::new(1));
        let current_impl_trait_defs = std::mem::take(&mut self.impl_trait_defs);
        let current_impl_trait_bounds = std::mem::take(&mut self.impl_trait_bounds);

        // Do not reset `next_node_id` and `node_id_to_def_id`:
        // we want `f` to be able to refer to the `LocalDefId`s that the caller created.
        // and the caller to refer to some of the subdefinitions' nodes' `LocalDefId`s.

        // Always allocate the first `HirId` for the owner itself.
        let _old = self.node_id_to_local_id.insert(owner, hir::ItemLocalId::new(0));
        debug_assert_eq!(_old, None);

        let item = f(self);
        debug_assert_eq!(def_id, item.def_id().def_id);
        // `f` should have consumed all the elements in these vectors when constructing `item`.
        debug_assert!(self.impl_trait_defs.is_empty());
        debug_assert!(self.impl_trait_bounds.is_empty());
        let info = self.make_owner_info(item);

        self.attrs = current_attrs;
        self.bodies = current_bodies;
        self.node_id_to_local_id = current_node_ids;
        self.trait_map = current_trait_map;
        self.current_hir_id_owner = current_owner;
        self.item_local_id_counter = current_local_counter;
        self.impl_trait_defs = current_impl_trait_defs;
        self.impl_trait_bounds = current_impl_trait_bounds;

        debug_assert!(!self.children.iter().any(|(id, _)| id == &def_id));
        self.children.push((def_id, hir::MaybeOwner::Owner(info)));
    }

    /// Installs the remapping `remap` in scope while `f` is being executed.
    /// This causes references to the `LocalDefId` keys to be changed to
    /// refer to the values instead.
    ///
    /// The remapping is used when one piece of AST expands to multiple
    /// pieces of HIR. For example, the function `fn foo<'a>(...) -> impl Debug + 'a`,
    /// expands to both a function definition (`foo`) and a TAIT for the return value,
    /// both of which have a lifetime parameter `'a`. The remapping allows us to
    /// rewrite the `'a` in the return value to refer to the
    /// `'a` declared on the TAIT, instead of the function.
    fn with_remapping<R>(
        &mut self,
        remap: FxHashMap<LocalDefId, LocalDefId>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.generics_def_id_map.push(remap);
        let res = f(self);
        self.generics_def_id_map.pop();
        res
    }

    fn make_owner_info(&mut self, node: hir::OwnerNode<'hir>) -> &'hir hir::OwnerInfo<'hir> {
        let attrs = std::mem::take(&mut self.attrs);
        let mut bodies = std::mem::take(&mut self.bodies);
        let trait_map = std::mem::take(&mut self.trait_map);

        #[cfg(debug_assertions)]
        for (id, attrs) in attrs.iter() {
            // Verify that we do not store empty slices in the map.
            if attrs.is_empty() {
                panic!("Stored empty attributes for {:?}", id);
            }
        }

        bodies.sort_by_key(|(k, _)| *k);
        let bodies = SortedMap::from_presorted_elements(bodies);

        // Don't hash unless necessary, because it's expensive.
        let (opt_hash_including_bodies, attrs_hash) = if self.tcx.needs_crate_hash() {
            self.tcx.with_stable_hashing_context(|mut hcx| {
                let mut stable_hasher = StableHasher::new();
                hcx.with_hir_bodies(node.def_id(), &bodies, |hcx| {
                    node.hash_stable(hcx, &mut stable_hasher)
                });
                let h1 = stable_hasher.finish();

                let mut stable_hasher = StableHasher::new();
                attrs.hash_stable(&mut hcx, &mut stable_hasher);
                let h2 = stable_hasher.finish();

                (Some(h1), Some(h2))
            })
        } else {
            (None, None)
        };
        let (nodes, parenting) = index::index_hir(self.tcx, node, &bodies);
        let nodes = hir::OwnerNodes { opt_hash_including_bodies, nodes, bodies };
        let attrs = hir::AttributeMap { map: attrs, opt_hash: attrs_hash };

        self.arena.alloc(hir::OwnerInfo { nodes, parenting, attrs, trait_map })
    }

    /// This method allocates a new `HirId` for the given `NodeId` and stores it in
    /// the `LoweringContext`'s `NodeId => HirId` map.
    /// Take care not to call this method if the resulting `HirId` is then not
    /// actually used in the HIR, as that would trigger an assertion in the
    /// `HirIdValidator` later on, which makes sure that all `NodeId`s got mapped
    /// properly. Calling the method twice with the same `NodeId` is fine though.
    #[instrument(level = "debug", skip(self), ret)]
    fn lower_node_id(&mut self, ast_node_id: NodeId) -> hir::HirId {
        assert_ne!(ast_node_id, DUMMY_NODE_ID);

        match self.node_id_to_local_id.entry(ast_node_id) {
            Entry::Occupied(o) => {
                hir::HirId { owner: self.current_hir_id_owner, local_id: *o.get() }
            }
            Entry::Vacant(v) => {
                // Generate a new `HirId`.
                let owner = self.current_hir_id_owner;
                let local_id = self.item_local_id_counter;
                let hir_id = hir::HirId { owner, local_id };

                v.insert(local_id);
                self.item_local_id_counter.increment_by(1);

                assert_ne!(local_id, hir::ItemLocalId::new(0));
                if let Some(def_id) = self.opt_local_def_id(ast_node_id) {
                    self.children.push((def_id, hir::MaybeOwner::NonOwner(hir_id)));
                }

                if let Some(traits) = self.resolver.trait_map.remove(&ast_node_id) {
                    self.trait_map.insert(hir_id.local_id, traits.into_boxed_slice());
                }

                hir_id
            }
        }
    }

    /// Generate a new `HirId` without a backing `NodeId`.
    #[instrument(level = "debug", skip(self), ret)]
    fn next_id(&mut self) -> hir::HirId {
        let owner = self.current_hir_id_owner;
        let local_id = self.item_local_id_counter;
        assert_ne!(local_id, hir::ItemLocalId::new(0));
        self.item_local_id_counter.increment_by(1);
        hir::HirId { owner, local_id }
    }

    #[instrument(level = "trace", skip(self))]
    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        let res: Result<Res, ()> = res.apply_id(|id| {
            let owner = self.current_hir_id_owner;
            let local_id = self.node_id_to_local_id.get(&id).copied().ok_or(())?;
            Ok(hir::HirId { owner, local_id })
        });
        trace!(?res);

        // We may fail to find a HirId when the Res points to a Local from an enclosing HIR owner.
        // This can happen when trying to lower the return type `x` in erroneous code like
        //   async fn foo(x: u8) -> x {}
        // In that case, `x` is lowered as a function parameter, and the return type is lowered as
        // an opaque type as a synthesized HIR owner.
        res.unwrap_or(Res::Err)
    }

    fn expect_full_res(&mut self, id: NodeId) -> Res<NodeId> {
        self.resolver.get_partial_res(id).map_or(Res::Err, |pr| pr.expect_full_res())
    }

    fn expect_full_res_from_use(&mut self, id: NodeId) -> impl Iterator<Item = Res<NodeId>> {
        self.resolver.get_import_res(id).present_items()
    }

    fn diagnostic(&self) -> &Handler {
        self.tcx.sess.diagnostic()
    }

    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    fn mark_span_with_reason(
        &self,
        reason: DesugaringKind,
        span: Span,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
    ) -> Span {
        self.tcx.with_stable_hashing_context(|hcx| {
            span.mark_with_reason(allow_internal_unstable, reason, self.tcx.sess.edition(), hcx)
        })
    }

    /// Intercept all spans entering HIR.
    /// Mark a span as relative to the current owning item.
    fn lower_span(&self, span: Span) -> Span {
        if self.tcx.sess.opts.incremental.is_some() {
            span.with_parent(Some(self.current_hir_id_owner.def_id))
        } else {
            // Do not make spans relative when not using incremental compilation.
            span
        }
    }

    fn lower_ident(&self, ident: Ident) -> Ident {
        Ident::new(ident.name, self.lower_span(ident.span))
    }

    /// Converts a lifetime into a new generic parameter.
    #[instrument(level = "debug", skip(self))]
    fn lifetime_res_to_generic_param(
        &mut self,
        ident: Ident,
        node_id: NodeId,
        res: LifetimeRes,
        source: hir::GenericParamSource,
    ) -> Option<hir::GenericParam<'hir>> {
        let (name, kind) = match res {
            LifetimeRes::Param { .. } => {
                (hir::ParamName::Plain(ident), hir::LifetimeParamKind::Explicit)
            }
            LifetimeRes::Fresh { param, .. } => {
                // Late resolution delegates to us the creation of the `LocalDefId`.
                let _def_id = self.create_def(
                    self.current_hir_id_owner.def_id,
                    param,
                    DefPathData::LifetimeNs(kw::UnderscoreLifetime),
                    ident.span,
                );
                debug!(?_def_id);

                (hir::ParamName::Fresh, hir::LifetimeParamKind::Elided)
            }
            LifetimeRes::Static | LifetimeRes::Error => return None,
            res => panic!(
                "Unexpected lifetime resolution {:?} for {:?} at {:?}",
                res, ident, ident.span
            ),
        };
        let hir_id = self.lower_node_id(node_id);
        let def_id = self.local_def_id(node_id);
        Some(hir::GenericParam {
            hir_id,
            def_id,
            name,
            span: self.lower_span(ident.span),
            pure_wrt_drop: false,
            kind: hir::GenericParamKind::Lifetime { kind },
            colon_span: None,
            source,
        })
    }

    /// Lowers a lifetime binder that defines `generic_params`, returning the corresponding HIR
    /// nodes. The returned list includes any "extra" lifetime parameters that were added by the
    /// name resolver owing to lifetime elision; this also populates the resolver's node-id->def-id
    /// map, so that later calls to `opt_node_id_to_def_id` that refer to these extra lifetime
    /// parameters will be successful.
    #[instrument(level = "debug", skip(self))]
    #[inline]
    fn lower_lifetime_binder(
        &mut self,
        binder: NodeId,
        generic_params: &[GenericParam],
    ) -> &'hir [hir::GenericParam<'hir>] {
        let mut generic_params: Vec<_> = self
            .lower_generic_params_mut(generic_params, hir::GenericParamSource::Binder)
            .collect();
        let extra_lifetimes = self.resolver.take_extra_lifetime_params(binder);
        debug!(?extra_lifetimes);
        generic_params.extend(extra_lifetimes.into_iter().filter_map(|(ident, node_id, res)| {
            self.lifetime_res_to_generic_param(ident, node_id, res, hir::GenericParamSource::Binder)
        }));
        let generic_params = self.arena.alloc_from_iter(generic_params);
        debug!(?generic_params);

        generic_params
    }

    fn with_dyn_type_scope<T>(&mut self, in_scope: bool, f: impl FnOnce(&mut Self) -> T) -> T {
        let was_in_dyn_type = self.is_in_dyn_type;
        self.is_in_dyn_type = in_scope;

        let result = f(self);

        self.is_in_dyn_type = was_in_dyn_type;

        result
    }

    fn with_new_scopes<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let catch_scope = self.catch_scope.take();
        let loop_scope = self.loop_scope.take();
        let ret = f(self);
        self.catch_scope = catch_scope;
        self.loop_scope = loop_scope;

        self.is_in_loop_condition = was_in_loop_condition;

        ret
    }

    fn lower_attrs(&mut self, id: hir::HirId, attrs: &[Attribute]) -> Option<&'hir [Attribute]> {
        if attrs.is_empty() {
            None
        } else {
            debug_assert_eq!(id.owner, self.current_hir_id_owner);
            let ret = self.arena.alloc_from_iter(attrs.iter().map(|a| self.lower_attr(a)));
            debug_assert!(!ret.is_empty());
            self.attrs.insert(id.local_id, ret);
            Some(ret)
        }
    }

    fn lower_attr(&self, attr: &Attribute) -> Attribute {
        // Note that we explicitly do not walk the path. Since we don't really
        // lower attributes (we use the AST version) there is nowhere to keep
        // the `HirId`s. We don't actually need HIR version of attributes anyway.
        // Tokens are also not needed after macro expansion and parsing.
        let kind = match attr.kind {
            AttrKind::Normal(ref normal) => AttrKind::Normal(P(NormalAttr {
                item: AttrItem {
                    path: normal.item.path.clone(),
                    args: self.lower_attr_args(&normal.item.args),
                    tokens: None,
                },
                tokens: None,
            })),
            AttrKind::DocComment(comment_kind, data) => AttrKind::DocComment(comment_kind, data),
        };

        Attribute { kind, id: attr.id, style: attr.style, span: self.lower_span(attr.span) }
    }

    fn alias_attrs(&mut self, id: hir::HirId, target_id: hir::HirId) {
        debug_assert_eq!(id.owner, self.current_hir_id_owner);
        debug_assert_eq!(target_id.owner, self.current_hir_id_owner);
        if let Some(&a) = self.attrs.get(&target_id.local_id) {
            debug_assert!(!a.is_empty());
            self.attrs.insert(id.local_id, a);
        }
    }

    fn lower_attr_args(&self, args: &AttrArgs) -> AttrArgs {
        match args {
            AttrArgs::Empty => AttrArgs::Empty,
            AttrArgs::Delimited(args) => AttrArgs::Delimited(self.lower_delim_args(args)),
            // This is an inert key-value attribute - it will never be visible to macros
            // after it gets lowered to HIR. Therefore, we can extract literals to handle
            // nonterminals in `#[doc]` (e.g. `#[doc = $e]`).
            AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => {
                // In valid code the value always ends up as a single literal. Otherwise, a dummy
                // literal suffices because the error is handled elsewhere.
                let lit = if let ExprKind::Lit(token_lit) = expr.kind
                    && let Ok(lit) = MetaItemLit::from_token_lit(token_lit, expr.span)
                {
                    lit
                } else {
                    MetaItemLit {
                        symbol: kw::Empty,
                        suffix: None,
                        kind: LitKind::Err,
                        span: DUMMY_SP,
                    }
                };
                AttrArgs::Eq(*eq_span, AttrArgsEq::Hir(lit))
            }
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when lowering mac args eq: {:?}", lit)
            }
        }
    }

    fn lower_delim_args(&self, args: &DelimArgs) -> DelimArgs {
        DelimArgs { dspan: args.dspan, delim: args.delim, tokens: args.tokens.flattened() }
    }

    /// Given an associated type constraint like one of these:
    ///
    /// ```ignore (illustrative)
    /// T: Iterator<Item: Debug>
    ///             ^^^^^^^^^^^
    /// T: Iterator<Item = Debug>
    ///             ^^^^^^^^^^^^
    /// ```
    ///
    /// returns a `hir::TypeBinding` representing `Item`.
    #[instrument(level = "debug", skip(self))]
    fn lower_assoc_ty_constraint(
        &mut self,
        constraint: &AssocConstraint,
        itctx: &ImplTraitContext,
    ) -> hir::TypeBinding<'hir> {
        debug!("lower_assoc_ty_constraint(constraint={:?}, itctx={:?})", constraint, itctx);
        // lower generic arguments of identifier in constraint
        let gen_args = if let Some(gen_args) = &constraint.gen_args {
            let gen_args_ctor = match gen_args {
                GenericArgs::AngleBracketed(data) => {
                    self.lower_angle_bracketed_parameter_data(data, ParamMode::Explicit, itctx).0
                }
                GenericArgs::Parenthesized(data) => {
                    if data.inputs.is_empty() && matches!(data.output, FnRetTy::Default(..)) {
                        let parenthesized = if self.tcx.features().return_type_notation {
                            hir::GenericArgsParentheses::ReturnTypeNotation
                        } else {
                            self.emit_bad_parenthesized_trait_in_assoc_ty(data);
                            hir::GenericArgsParentheses::No
                        };
                        GenericArgsCtor {
                            args: Default::default(),
                            bindings: &[],
                            parenthesized,
                            span: data.inputs_span,
                        }
                    } else if let Some(first_char) = constraint.ident.as_str().chars().next()
                        && first_char.is_ascii_lowercase()
                    {
                        let mut err = if !data.inputs.is_empty() {
                            self.tcx.sess.create_err(errors::BadReturnTypeNotation::Inputs {
                                span: data.inputs_span,
                            })
                        } else if let FnRetTy::Ty(ty) = &data.output {
                            self.tcx.sess.create_err(errors::BadReturnTypeNotation::Output {
                                span: data.inputs_span.shrink_to_hi().to(ty.span),
                            })
                        } else {
                            unreachable!("inputs are empty and return type is not provided")
                        };
                        if !self.tcx.features().return_type_notation
                            && self.tcx.sess.is_nightly_build()
                        {
                            add_feature_diagnostics(
                                &mut err,
                                &self.tcx.sess.parse_sess,
                                sym::return_type_notation,
                            );
                        }
                        err.emit();
                        GenericArgsCtor {
                            args: Default::default(),
                            bindings: &[],
                            parenthesized: hir::GenericArgsParentheses::ReturnTypeNotation,
                            span: data.span,
                        }
                    } else {
                        self.emit_bad_parenthesized_trait_in_assoc_ty(data);
                        // FIXME(return_type_notation): we could issue a feature error
                        // if the parens are empty and there's no return type.
                        self.lower_angle_bracketed_parameter_data(
                            &data.as_angle_bracketed_args(),
                            ParamMode::Explicit,
                            itctx,
                        )
                        .0
                    }
                }
            };
            gen_args_ctor.into_generic_args(self)
        } else {
            self.arena.alloc(hir::GenericArgs::none())
        };
        let kind = match &constraint.kind {
            AssocConstraintKind::Equality { term } => {
                let term = match term {
                    Term::Ty(ty) => self.lower_ty(ty, itctx).into(),
                    Term::Const(c) => self.lower_anon_const(c).into(),
                };
                hir::TypeBindingKind::Equality { term }
            }
            AssocConstraintKind::Bound { bounds } => {
                enum DesugarKind<'a> {
                    ImplTrait,
                    Error(&'a ImplTraitPosition),
                    Bound,
                }

                // Piggy-back on the `impl Trait` context to figure out the correct behavior.
                let desugar_kind = match itctx {
                    // We are in the return position:
                    //
                    //     fn foo() -> impl Iterator<Item: Debug>
                    //
                    // so desugar to
                    //
                    //     fn foo() -> impl Iterator<Item = impl Debug>
                    ImplTraitContext::ReturnPositionOpaqueTy { .. }
                    | ImplTraitContext::TypeAliasesOpaqueTy { .. } => DesugarKind::ImplTrait,

                    // We are in the argument position, but within a dyn type:
                    //
                    //     fn foo(x: dyn Iterator<Item: Debug>)
                    //
                    // so desugar to
                    //
                    //     fn foo(x: dyn Iterator<Item = impl Debug>)
                    ImplTraitContext::Universal if self.is_in_dyn_type => DesugarKind::ImplTrait,

                    ImplTraitContext::Disallowed(position) if self.is_in_dyn_type => {
                        DesugarKind::Error(position)
                    }

                    // We are in the parameter position, but not within a dyn type:
                    //
                    //     fn foo(x: impl Iterator<Item: Debug>)
                    //
                    // so we leave it as is and this gets expanded in astconv to a bound like
                    // `<T as Iterator>::Item: Debug` where `T` is the type parameter for the
                    // `impl Iterator`.
                    _ => DesugarKind::Bound,
                };

                match desugar_kind {
                    DesugarKind::ImplTrait => {
                        // Desugar `AssocTy: Bounds` into `AssocTy = impl Bounds`. We do this by
                        // constructing the HIR for `impl bounds...` and then lowering that.

                        let impl_trait_node_id = self.next_node_id();
                        // Shift `impl Trait` lifetime captures from the associated type bound's
                        // node id to the opaque node id, so that the opaque can actually use
                        // these lifetime bounds.
                        self.resolver
                            .remap_extra_lifetime_params(constraint.id, impl_trait_node_id);

                        self.with_dyn_type_scope(false, |this| {
                            let node_id = this.next_node_id();
                            let ty = this.lower_ty(
                                &Ty {
                                    id: node_id,
                                    kind: TyKind::ImplTrait(impl_trait_node_id, bounds.clone()),
                                    span: this.lower_span(constraint.span),
                                    tokens: None,
                                },
                                itctx,
                            );

                            hir::TypeBindingKind::Equality { term: ty.into() }
                        })
                    }
                    DesugarKind::Bound => {
                        // Desugar `AssocTy: Bounds` into a type binding where the
                        // later desugars into a trait predicate.
                        let bounds = self.lower_param_bounds(bounds, itctx);

                        hir::TypeBindingKind::Constraint { bounds }
                    }
                    DesugarKind::Error(position) => {
                        let guar = self.tcx.sess.emit_err(errors::MisplacedAssocTyBinding {
                            span: constraint.span,
                            position: DiagnosticArgFromDisplay(position),
                        });
                        let err_ty =
                            &*self.arena.alloc(self.ty(constraint.span, hir::TyKind::Err(guar)));
                        hir::TypeBindingKind::Equality { term: err_ty.into() }
                    }
                }
            }
        };

        hir::TypeBinding {
            hir_id: self.lower_node_id(constraint.id),
            ident: self.lower_ident(constraint.ident),
            gen_args,
            kind,
            span: self.lower_span(constraint.span),
        }
    }

    fn emit_bad_parenthesized_trait_in_assoc_ty(&self, data: &ParenthesizedArgs) {
        // Suggest removing empty parentheses: "Trait()" -> "Trait"
        let sub = if data.inputs.is_empty() {
            let parentheses_span =
                data.inputs_span.shrink_to_lo().to(data.inputs_span.shrink_to_hi());
            AssocTyParenthesesSub::Empty { parentheses_span }
        }
        // Suggest replacing parentheses with angle brackets `Trait(params...)` to `Trait<params...>`
        else {
            // Start of parameters to the 1st argument
            let open_param = data.inputs_span.shrink_to_lo().to(data
                .inputs
                .first()
                .unwrap()
                .span
                .shrink_to_lo());
            // End of last argument to end of parameters
            let close_param =
                data.inputs.last().unwrap().span.shrink_to_hi().to(data.inputs_span.shrink_to_hi());
            AssocTyParenthesesSub::NotEmpty { open_param, close_param }
        };
        self.tcx.sess.emit_err(AssocTyParentheses { span: data.span, sub });
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_generic_arg(
        &mut self,
        arg: &ast::GenericArg,
        itctx: &ImplTraitContext,
    ) -> hir::GenericArg<'hir> {
        match arg {
            ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.lower_lifetime(&lt)),
            ast::GenericArg::Type(ty) => {
                match &ty.kind {
                    TyKind::Infer if self.tcx.features().generic_arg_infer => {
                        return GenericArg::Infer(hir::InferArg {
                            hir_id: self.lower_node_id(ty.id),
                            span: self.lower_span(ty.span),
                        });
                    }
                    // We parse const arguments as path types as we cannot distinguish them during
                    // parsing. We try to resolve that ambiguity by attempting resolution in both the
                    // type and value namespaces. If we resolved the path in the value namespace, we
                    // transform it into a generic const argument.
                    TyKind::Path(None, path) => {
                        if let Some(res) = self
                            .resolver
                            .get_partial_res(ty.id)
                            .and_then(|partial_res| partial_res.full_res())
                        {
                            if !res.matches_ns(Namespace::TypeNS)
                                && path.is_potential_trivial_const_arg()
                            {
                                debug!(
                                    "lower_generic_arg: Lowering type argument as const argument: {:?}",
                                    ty,
                                );

                                // Construct an AnonConst where the expr is the "ty"'s path.

                                let parent_def_id = self.current_hir_id_owner;
                                let node_id = self.next_node_id();
                                let span = self.lower_span(ty.span);

                                // Add a definition for the in-band const def.
                                let def_id = self.create_def(
                                    parent_def_id.def_id,
                                    node_id,
                                    DefPathData::AnonConst,
                                    span,
                                );

                                let path_expr = Expr {
                                    id: ty.id,
                                    kind: ExprKind::Path(None, path.clone()),
                                    span,
                                    attrs: AttrVec::new(),
                                    tokens: None,
                                };

                                let ct = self.with_new_scopes(|this| hir::AnonConst {
                                    def_id,
                                    hir_id: this.lower_node_id(node_id),
                                    body: this.lower_const_body(path_expr.span, Some(&path_expr)),
                                });
                                return GenericArg::Const(ConstArg { value: ct, span });
                            }
                        }
                    }
                    _ => {}
                }
                GenericArg::Type(self.lower_ty(&ty, itctx))
            }
            ast::GenericArg::Const(ct) => GenericArg::Const(ConstArg {
                value: self.lower_anon_const(&ct),
                span: self.lower_span(ct.value.span),
            }),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_ty(&mut self, t: &Ty, itctx: &ImplTraitContext) -> &'hir hir::Ty<'hir> {
        self.arena.alloc(self.lower_ty_direct(t, itctx))
    }

    fn lower_path_ty(
        &mut self,
        t: &Ty,
        qself: &Option<ptr::P<QSelf>>,
        path: &Path,
        param_mode: ParamMode,
        itctx: &ImplTraitContext,
    ) -> hir::Ty<'hir> {
        // Check whether we should interpret this as a bare trait object.
        // This check mirrors the one in late resolution. We only introduce this special case in
        // the rare occurrence we need to lower `Fresh` anonymous lifetimes.
        // The other cases when a qpath should be opportunistically made a trait object are handled
        // by `ty_path`.
        if qself.is_none()
            && let Some(partial_res) = self.resolver.get_partial_res(t.id)
            && let Some(Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) = partial_res.full_res()
        {
            let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                let bound = this.lower_poly_trait_ref(
                    &PolyTraitRef {
                        bound_generic_params: ThinVec::new(),
                        trait_ref: TraitRef { path: path.clone(), ref_id: t.id },
                        span: t.span,
                    },
                    itctx,
                    ast::Const::No,
                );
                let bounds = this.arena.alloc_from_iter([bound]);
                let lifetime_bound = this.elided_dyn_bound(t.span);
                (bounds, lifetime_bound)
            });
            let kind = hir::TyKind::TraitObject(bounds, &lifetime_bound, TraitObjectSyntax::None);
            return hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.next_id() };
        }

        let id = self.lower_node_id(t.id);
        let qpath = self.lower_qpath(t.id, qself, path, param_mode, itctx, None);
        self.ty_path(id, t.span, qpath)
    }

    fn ty(&mut self, span: Span, kind: hir::TyKind<'hir>) -> hir::Ty<'hir> {
        hir::Ty { hir_id: self.next_id(), kind, span: self.lower_span(span) }
    }

    fn ty_tup(&mut self, span: Span, tys: &'hir [hir::Ty<'hir>]) -> hir::Ty<'hir> {
        self.ty(span, hir::TyKind::Tup(tys))
    }

    fn lower_ty_direct(&mut self, t: &Ty, itctx: &ImplTraitContext) -> hir::Ty<'hir> {
        let kind = match &t.kind {
            TyKind::Infer => hir::TyKind::Infer,
            TyKind::Err => {
                hir::TyKind::Err(self.tcx.sess.delay_span_bug(t.span, "TyKind::Err lowered"))
            }
            // FIXME(unnamed_fields): IMPLEMENTATION IN PROGRESS
            #[allow(rustc::untranslatable_diagnostic)]
            #[allow(rustc::diagnostic_outside_of_impl)]
            TyKind::AnonStruct(ref _fields) => hir::TyKind::Err(
                self.tcx.sess.span_err(t.span, "anonymous structs are unimplemented"),
            ),
            // FIXME(unnamed_fields): IMPLEMENTATION IN PROGRESS
            #[allow(rustc::untranslatable_diagnostic)]
            #[allow(rustc::diagnostic_outside_of_impl)]
            TyKind::AnonUnion(ref _fields) => hir::TyKind::Err(
                self.tcx.sess.span_err(t.span, "anonymous unions are unimplemented"),
            ),
            TyKind::Slice(ty) => hir::TyKind::Slice(self.lower_ty(ty, itctx)),
            TyKind::Ptr(mt) => hir::TyKind::Ptr(self.lower_mt(mt, itctx)),
            TyKind::Ref(region, mt) => {
                let region = region.unwrap_or_else(|| {
                    let id = if let Some(LifetimeRes::ElidedAnchor { start, end }) =
                        self.resolver.get_lifetime_res(t.id)
                    {
                        debug_assert_eq!(start.plus(1), end);
                        start
                    } else {
                        self.next_node_id()
                    };
                    let span = self.tcx.sess.source_map().start_point(t.span).shrink_to_hi();
                    Lifetime { ident: Ident::new(kw::UnderscoreLifetime, span), id }
                });
                let lifetime = self.lower_lifetime(&region);
                hir::TyKind::Ref(lifetime, self.lower_mt(mt, itctx))
            }
            TyKind::BareFn(f) => {
                let generic_params = self.lower_lifetime_binder(t.id, &f.generic_params);
                hir::TyKind::BareFn(self.arena.alloc(hir::BareFnTy {
                    generic_params,
                    unsafety: self.lower_unsafety(f.unsafety),
                    abi: self.lower_extern(f.ext),
                    decl: self.lower_fn_decl(&f.decl, t.id, t.span, FnDeclKind::Pointer, None),
                    param_names: self.lower_fn_params_to_names(&f.decl),
                }))
            }
            TyKind::Never => hir::TyKind::Never,
            TyKind::Tup(tys) => hir::TyKind::Tup(
                self.arena.alloc_from_iter(tys.iter().map(|ty| self.lower_ty_direct(ty, itctx))),
            ),
            TyKind::Paren(ty) => {
                return self.lower_ty_direct(ty, itctx);
            }
            TyKind::Path(qself, path) => {
                return self.lower_path_ty(t, qself, path, ParamMode::Explicit, itctx);
            }
            TyKind::ImplicitSelf => {
                let hir_id = self.next_id();
                let res = self.expect_full_res(t.id);
                let res = self.lower_res(res);
                hir::TyKind::Path(hir::QPath::Resolved(
                    None,
                    self.arena.alloc(hir::Path {
                        res,
                        segments: arena_vec![self; hir::PathSegment::new(
                            Ident::with_dummy_span(kw::SelfUpper),
                            hir_id,
                            res
                        )],
                        span: self.lower_span(t.span),
                    }),
                ))
            }
            TyKind::Array(ty, length) => {
                hir::TyKind::Array(self.lower_ty(ty, itctx), self.lower_array_length(length))
            }
            TyKind::Typeof(expr) => hir::TyKind::Typeof(self.lower_anon_const(expr)),
            TyKind::TraitObject(bounds, kind) => {
                let mut lifetime_bound = None;
                let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                    let bounds =
                        this.arena.alloc_from_iter(bounds.iter().filter_map(|bound| match bound {
                            GenericBound::Trait(
                                ty,
                                modifier @ (TraitBoundModifier::None
                                | TraitBoundModifier::MaybeConst
                                | TraitBoundModifier::Negative),
                            ) => {
                                Some(this.lower_poly_trait_ref(ty, itctx, modifier.to_constness()))
                            }
                            // `~const ?Bound` will cause an error during AST validation
                            // anyways, so treat it like `?Bound` as compilation proceeds.
                            GenericBound::Trait(
                                _,
                                TraitBoundModifier::Maybe
                                | TraitBoundModifier::MaybeConstMaybe
                                | TraitBoundModifier::MaybeConstNegative,
                            ) => None,
                            GenericBound::Outlives(lifetime) => {
                                if lifetime_bound.is_none() {
                                    lifetime_bound = Some(this.lower_lifetime(lifetime));
                                }
                                None
                            }
                        }));
                    let lifetime_bound =
                        lifetime_bound.unwrap_or_else(|| this.elided_dyn_bound(t.span));
                    (bounds, lifetime_bound)
                });
                hir::TyKind::TraitObject(bounds, lifetime_bound, *kind)
            }
            TyKind::ImplTrait(def_node_id, bounds) => {
                let span = t.span;
                match itctx {
                    ImplTraitContext::ReturnPositionOpaqueTy { origin, fn_kind } => self
                        .lower_opaque_impl_trait(
                            span,
                            *origin,
                            *def_node_id,
                            bounds,
                            Some(*fn_kind),
                            itctx,
                        ),
                    &ImplTraitContext::TypeAliasesOpaqueTy { in_assoc_ty } => self
                        .lower_opaque_impl_trait(
                            span,
                            hir::OpaqueTyOrigin::TyAlias { in_assoc_ty },
                            *def_node_id,
                            bounds,
                            None,
                            itctx,
                        ),
                    ImplTraitContext::Universal => {
                        let span = t.span;

                        // HACK: pprust breaks strings with newlines when the type
                        // gets too long. We don't want these to show up in compiler
                        // output or built artifacts, so replace them here...
                        // Perhaps we should instead format APITs more robustly.
                        let ident = Ident::from_str_and_span(
                            &pprust::ty_to_string(t).replace('\n', " "),
                            span,
                        );

                        self.create_def(
                            self.current_hir_id_owner.def_id,
                            *def_node_id,
                            DefPathData::TypeNs(ident.name),
                            span,
                        );
                        let (param, bounds, path) = self.lower_universal_param_and_bounds(
                            *def_node_id,
                            span,
                            ident,
                            bounds,
                        );
                        self.impl_trait_defs.push(param);
                        if let Some(bounds) = bounds {
                            self.impl_trait_bounds.push(bounds);
                        }
                        path
                    }
                    ImplTraitContext::FeatureGated(position, feature) => {
                        let guar = self
                            .tcx
                            .sess
                            .create_feature_err(
                                MisplacedImplTrait {
                                    span: t.span,
                                    position: DiagnosticArgFromDisplay(position),
                                },
                                *feature,
                            )
                            .emit();
                        hir::TyKind::Err(guar)
                    }
                    ImplTraitContext::Disallowed(position) => {
                        let guar = self.tcx.sess.emit_err(MisplacedImplTrait {
                            span: t.span,
                            position: DiagnosticArgFromDisplay(position),
                        });
                        hir::TyKind::Err(guar)
                    }
                }
            }
            TyKind::MacCall(_) => panic!("`TyKind::MacCall` should have been expanded by now"),
            TyKind::CVarArgs => {
                let guar = self.tcx.sess.delay_span_bug(
                    t.span,
                    "`TyKind::CVarArgs` should have been handled elsewhere",
                );
                hir::TyKind::Err(guar)
            }
        };

        hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.lower_node_id(t.id) }
    }

    /// Lowers a `ReturnPositionOpaqueTy` (`-> impl Trait`) or a `TypeAliasesOpaqueTy` (`type F =
    /// impl Trait`): this creates the associated Opaque Type (TAIT) definition and then returns a
    /// HIR type that references the TAIT.
    ///
    /// Given a function definition like:
    ///
    /// ```rust
    /// use std::fmt::Debug;
    ///
    /// fn test<'a, T: Debug>(x: &'a T) -> impl Debug + 'a {
    ///     x
    /// }
    /// ```
    ///
    /// we will create a TAIT definition in the HIR like
    ///
    /// ```rust,ignore (pseudo-Rust)
    /// type TestReturn<'a, T, 'x> = impl Debug + 'x
    /// ```
    ///
    /// and return a type like `TestReturn<'static, T, 'a>`, so that the function looks like:
    ///
    /// ```rust,ignore (pseudo-Rust)
    /// fn test<'a, T: Debug>(x: &'a T) -> TestReturn<'static, T, 'a>
    /// ```
    ///
    /// Note the subtlety around type parameters! The new TAIT, `TestReturn`, inherits all the
    /// type parameters from the function `test` (this is implemented in the query layer, they aren't
    /// added explicitly in the HIR). But this includes all the lifetimes, and we only want to
    /// capture the lifetimes that are referenced in the bounds. Therefore, we add *extra* lifetime parameters
    /// for the lifetimes that get captured (`'x`, in our example above) and reference those.
    #[instrument(level = "debug", skip(self), ret)]
    fn lower_opaque_impl_trait(
        &mut self,
        span: Span,
        origin: hir::OpaqueTyOrigin,
        opaque_ty_node_id: NodeId,
        bounds: &GenericBounds,
        fn_kind: Option<FnDeclKind>,
        itctx: &ImplTraitContext,
    ) -> hir::TyKind<'hir> {
        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::OpaqueTy, span, None);

        let captured_lifetimes_to_duplicate = match origin {
            hir::OpaqueTyOrigin::TyAlias { .. } => {
                // in a TAIT like `type Foo<'a> = impl Foo<'a>`, we don't duplicate any
                // lifetimes, since we don't have the issue that any are late-bound.
                Vec::new()
            }
            hir::OpaqueTyOrigin::FnReturn(..) => {
                if let FnDeclKind::Impl | FnDeclKind::Trait =
                    fn_kind.expect("expected RPITs to be lowered with a FnKind")
                {
                    // return-position impl trait in trait was decided to capture all
                    // in-scope lifetimes, which we collect for all opaques during resolution.
                    self.resolver
                        .take_extra_lifetime_params(opaque_ty_node_id)
                        .into_iter()
                        .map(|(ident, id, _)| Lifetime { id, ident })
                        .collect()
                } else {
                    // in fn return position, like the `fn test<'a>() -> impl Debug + 'a`
                    // example, we only need to duplicate lifetimes that appear in the
                    // bounds, since those are the only ones that are captured by the opaque.
                    lifetime_collector::lifetimes_in_bounds(&self.resolver, bounds)
                }
            }
            hir::OpaqueTyOrigin::AsyncFn(..) => {
                unreachable!("should be using `lower_async_fn_ret_ty`")
            }
        };
        debug!(?captured_lifetimes_to_duplicate);

        self.lower_opaque_inner(
            opaque_ty_node_id,
            origin,
            matches!(fn_kind, Some(FnDeclKind::Trait)),
            captured_lifetimes_to_duplicate,
            span,
            opaque_ty_span,
            |this| this.lower_param_bounds(bounds, itctx),
        )
    }

    fn lower_opaque_inner(
        &mut self,
        opaque_ty_node_id: NodeId,
        origin: hir::OpaqueTyOrigin,
        in_trait: bool,
        captured_lifetimes_to_duplicate: Vec<Lifetime>,
        span: Span,
        opaque_ty_span: Span,
        lower_item_bounds: impl FnOnce(&mut Self) -> &'hir [hir::GenericBound<'hir>],
    ) -> hir::TyKind<'hir> {
        let opaque_ty_def_id = self.create_def(
            self.current_hir_id_owner.def_id,
            opaque_ty_node_id,
            DefPathData::ImplTrait,
            opaque_ty_span,
        );
        debug!(?opaque_ty_def_id);

        // Map from captured (old) lifetime to synthetic (new) lifetime.
        // Used to resolve lifetimes in the bounds of the opaque.
        let mut captured_to_synthesized_mapping = FxHashMap::default();
        // List of (early-bound) synthetic lifetimes that are owned by the opaque.
        // This is used to create the `hir::Generics` owned by the opaque.
        let mut synthesized_lifetime_definitions = vec![];
        // Pairs of lifetime arg (that resolves to the captured lifetime)
        // and the def-id of the (early-bound) synthetic lifetime definition.
        // This is used both to create generics for the `TyKind::OpaqueDef` that
        // we return, and also as a captured lifetime mapping for RPITITs.
        let mut synthesized_lifetime_args = vec![];

        for lifetime in captured_lifetimes_to_duplicate {
            let res = self.resolver.get_lifetime_res(lifetime.id).unwrap_or(LifetimeRes::Error);
            let old_def_id = match res {
                LifetimeRes::Param { param: old_def_id, binder: _ } => old_def_id,

                LifetimeRes::Fresh { param, binder: _ } => {
                    debug_assert_eq!(lifetime.ident.name, kw::UnderscoreLifetime);
                    if let Some(old_def_id) = self.orig_opt_local_def_id(param) {
                        old_def_id
                    } else {
                        self.tcx
                            .sess
                            .delay_span_bug(lifetime.ident.span, "no def-id for fresh lifetime");
                        continue;
                    }
                }

                // Opaques do not capture `'static`
                LifetimeRes::Static | LifetimeRes::Error => {
                    continue;
                }

                res => {
                    let bug_msg = format!(
                        "Unexpected lifetime resolution {:?} for {:?} at {:?}",
                        res, lifetime.ident, lifetime.ident.span
                    );
                    span_bug!(lifetime.ident.span, "{}", bug_msg);
                }
            };

            if captured_to_synthesized_mapping.get(&old_def_id).is_none() {
                // Create a new lifetime parameter local to the opaque.
                let duplicated_lifetime_node_id = self.next_node_id();
                let duplicated_lifetime_def_id = self.create_def(
                    opaque_ty_def_id,
                    duplicated_lifetime_node_id,
                    DefPathData::LifetimeNs(lifetime.ident.name),
                    lifetime.ident.span,
                );
                captured_to_synthesized_mapping.insert(old_def_id, duplicated_lifetime_def_id);
                // FIXME: Instead of doing this, we could move this whole loop
                // into the `with_hir_id_owner`, then just directly construct
                // the `hir::GenericParam` here.
                synthesized_lifetime_definitions.push((
                    duplicated_lifetime_node_id,
                    duplicated_lifetime_def_id,
                    lifetime.ident,
                ));

                // Now make an arg that we can use for the generic params of the opaque tykind.
                let id = self.next_node_id();
                let lifetime_arg = self.new_named_lifetime_with_res(id, lifetime.ident, res);
                let duplicated_lifetime_def_id = self.local_def_id(duplicated_lifetime_node_id);
                synthesized_lifetime_args.push((lifetime_arg, duplicated_lifetime_def_id))
            }
        }

        self.with_hir_id_owner(opaque_ty_node_id, |this| {
            // Install the remapping from old to new (if any). This makes sure that
            // any lifetimes that would have resolved to the def-id of captured
            // lifetimes are remapped to the new *synthetic* lifetimes of the opaque.
            let bounds = this
                .with_remapping(captured_to_synthesized_mapping, |this| lower_item_bounds(this));

            let generic_params = this.arena.alloc_from_iter(
                synthesized_lifetime_definitions.iter().map(|&(new_node_id, new_def_id, ident)| {
                    let hir_id = this.lower_node_id(new_node_id);
                    let (name, kind) = if ident.name == kw::UnderscoreLifetime {
                        (hir::ParamName::Fresh, hir::LifetimeParamKind::Elided)
                    } else {
                        (hir::ParamName::Plain(ident), hir::LifetimeParamKind::Explicit)
                    };

                    hir::GenericParam {
                        hir_id,
                        def_id: new_def_id,
                        name,
                        span: ident.span,
                        pure_wrt_drop: false,
                        kind: hir::GenericParamKind::Lifetime { kind },
                        colon_span: None,
                        source: hir::GenericParamSource::Generics,
                    }
                }),
            );
            debug!("lower_async_fn_ret_ty: generic_params={:#?}", generic_params);

            let lifetime_mapping = self.arena.alloc_slice(&synthesized_lifetime_args);

            let opaque_ty_item = hir::OpaqueTy {
                generics: this.arena.alloc(hir::Generics {
                    params: generic_params,
                    predicates: &[],
                    has_where_clause_predicates: false,
                    where_clause_span: this.lower_span(span),
                    span: this.lower_span(span),
                }),
                bounds,
                origin,
                lifetime_mapping,
                in_trait,
            };

            // Generate an `type Foo = impl Trait;` declaration.
            trace!("registering opaque type with id {:#?}", opaque_ty_def_id);
            let opaque_ty_item = hir::Item {
                owner_id: hir::OwnerId { def_id: opaque_ty_def_id },
                ident: Ident::empty(),
                kind: hir::ItemKind::OpaqueTy(this.arena.alloc(opaque_ty_item)),
                vis_span: this.lower_span(span.shrink_to_lo()),
                span: this.lower_span(opaque_ty_span),
            };

            hir::OwnerNode::Item(this.arena.alloc(opaque_ty_item))
        });

        let generic_args = self.arena.alloc_from_iter(
            synthesized_lifetime_args
                .iter()
                .map(|(lifetime, _)| hir::GenericArg::Lifetime(*lifetime)),
        );

        // Create the `Foo<...>` reference itself. Note that the `type
        // Foo = impl Trait` is, internally, created as a child of the
        // async fn, so the *type parameters* are inherited. It's
        // only the lifetime parameters that we must supply.
        hir::TyKind::OpaqueDef(
            hir::ItemId { owner_id: hir::OwnerId { def_id: opaque_ty_def_id } },
            generic_args,
            in_trait,
        )
    }

    fn lower_fn_params_to_names(&mut self, decl: &FnDecl) -> &'hir [Ident] {
        // Skip the `...` (`CVarArgs`) trailing arguments from the AST,
        // as they are not explicit in HIR/Ty function signatures.
        // (instead, the `c_variadic` flag is set to `true`)
        let mut inputs = &decl.inputs[..];
        if decl.c_variadic() {
            inputs = &inputs[..inputs.len() - 1];
        }
        self.arena.alloc_from_iter(inputs.iter().map(|param| match param.pat.kind {
            PatKind::Ident(_, ident, _) => self.lower_ident(ident),
            _ => Ident::new(kw::Empty, self.lower_span(param.pat.span)),
        }))
    }

    // Lowers a function declaration.
    //
    // `decl`: the unlowered (AST) function declaration.
    // `fn_node_id`: `impl Trait` arguments are lowered into generic parameters on the given `NodeId`.
    // `make_ret_async`: if `Some`, converts `-> T` into `-> impl Future<Output = T>` in the
    //      return type. This is used for `async fn` declarations. The `NodeId` is the ID of the
    //      return type `impl Trait` item, and the `Span` points to the `async` keyword.
    #[instrument(level = "debug", skip(self))]
    fn lower_fn_decl(
        &mut self,
        decl: &FnDecl,
        fn_node_id: NodeId,
        fn_span: Span,
        kind: FnDeclKind,
        make_ret_async: Option<(NodeId, Span)>,
    ) -> &'hir hir::FnDecl<'hir> {
        let c_variadic = decl.c_variadic();

        // Skip the `...` (`CVarArgs`) trailing arguments from the AST,
        // as they are not explicit in HIR/Ty function signatures.
        // (instead, the `c_variadic` flag is set to `true`)
        let mut inputs = &decl.inputs[..];
        if c_variadic {
            inputs = &inputs[..inputs.len() - 1];
        }
        let inputs = self.arena.alloc_from_iter(inputs.iter().map(|param| {
            let itctx = if kind.param_impl_trait_allowed() {
                ImplTraitContext::Universal
            } else {
                ImplTraitContext::Disallowed(match kind {
                    FnDeclKind::Fn | FnDeclKind::Inherent => {
                        unreachable!("fn should allow APIT")
                    }
                    FnDeclKind::ExternFn => ImplTraitPosition::ExternFnParam,
                    FnDeclKind::Closure => ImplTraitPosition::ClosureParam,
                    FnDeclKind::Pointer => ImplTraitPosition::PointerParam,
                    FnDeclKind::Trait => ImplTraitPosition::TraitParam,
                    FnDeclKind::Impl => ImplTraitPosition::ImplParam,
                })
            };
            self.lower_ty_direct(&param.ty, &itctx)
        }));

        let output = if let Some((ret_id, _span)) = make_ret_async {
            let fn_def_id = self.local_def_id(fn_node_id);
            self.lower_async_fn_ret_ty(&decl.output, fn_def_id, ret_id, kind, fn_span)
        } else {
            match &decl.output {
                FnRetTy::Ty(ty) => {
                    let context = if kind.return_impl_trait_allowed() {
                        let fn_def_id = self.local_def_id(fn_node_id);
                        ImplTraitContext::ReturnPositionOpaqueTy {
                            origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                            fn_kind: kind,
                        }
                    } else {
                        ImplTraitContext::Disallowed(match kind {
                            FnDeclKind::Fn
                            | FnDeclKind::Inherent
                            | FnDeclKind::Trait
                            | FnDeclKind::Impl => {
                                unreachable!("fn should allow return-position impl trait in traits")
                            }
                            FnDeclKind::ExternFn => ImplTraitPosition::ExternFnReturn,
                            FnDeclKind::Closure => ImplTraitPosition::ClosureReturn,
                            FnDeclKind::Pointer => ImplTraitPosition::PointerReturn,
                        })
                    };
                    hir::FnRetTy::Return(self.lower_ty(ty, &context))
                }
                FnRetTy::Default(span) => hir::FnRetTy::DefaultReturn(self.lower_span(*span)),
            }
        };

        self.arena.alloc(hir::FnDecl {
            inputs,
            output,
            c_variadic,
            lifetime_elision_allowed: self.resolver.lifetime_elision_allowed.contains(&fn_node_id),
            implicit_self: decl.inputs.get(0).map_or(hir::ImplicitSelfKind::None, |arg| {
                let is_mutable_pat = matches!(
                    arg.pat.kind,
                    PatKind::Ident(hir::BindingAnnotation(_, Mutability::Mut), ..)
                );

                match &arg.ty.kind {
                    TyKind::ImplicitSelf if is_mutable_pat => hir::ImplicitSelfKind::Mut,
                    TyKind::ImplicitSelf => hir::ImplicitSelfKind::Imm,
                    // Given we are only considering `ImplicitSelf` types, we needn't consider
                    // the case where we have a mutable pattern to a reference as that would
                    // no longer be an `ImplicitSelf`.
                    TyKind::Ref(_, mt) if mt.ty.kind.is_implicit_self() => match mt.mutbl {
                        hir::Mutability::Not => hir::ImplicitSelfKind::ImmRef,
                        hir::Mutability::Mut => hir::ImplicitSelfKind::MutRef,
                    },
                    _ => hir::ImplicitSelfKind::None,
                }
            }),
        })
    }

    // Transforms `-> T` for `async fn` into `-> OpaqueTy { .. }`
    // combined with the following definition of `OpaqueTy`:
    //
    //     type OpaqueTy<generics_from_parent_fn> = impl Future<Output = T>;
    //
    // `output`: unlowered output type (`T` in `-> T`)
    // `fn_node_id`: `NodeId` of the parent function (used to create child impl trait definition)
    // `opaque_ty_node_id`: `NodeId` of the opaque `impl Trait` type that should be created
    #[instrument(level = "debug", skip(self))]
    fn lower_async_fn_ret_ty(
        &mut self,
        output: &FnRetTy,
        fn_def_id: LocalDefId,
        opaque_ty_node_id: NodeId,
        fn_kind: FnDeclKind,
        fn_span: Span,
    ) -> hir::FnRetTy<'hir> {
        let span = self.lower_span(fn_span);
        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::Async, span, None);

        let captured_lifetimes: Vec<_> = self
            .resolver
            .take_extra_lifetime_params(opaque_ty_node_id)
            .into_iter()
            .map(|(ident, id, _)| Lifetime { id, ident })
            .collect();

        let opaque_ty_ref = self.lower_opaque_inner(
            opaque_ty_node_id,
            hir::OpaqueTyOrigin::AsyncFn(fn_def_id),
            matches!(fn_kind, FnDeclKind::Trait),
            captured_lifetimes,
            span,
            opaque_ty_span,
            |this| {
                let future_bound = this.lower_async_fn_output_type_to_future_bound(
                    output,
                    span,
                    ImplTraitContext::ReturnPositionOpaqueTy {
                        origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                        fn_kind,
                    },
                );
                arena_vec![this; future_bound]
            },
        );

        let opaque_ty = self.ty(opaque_ty_span, opaque_ty_ref);
        hir::FnRetTy::Return(self.arena.alloc(opaque_ty))
    }

    /// Transforms `-> T` into `Future<Output = T>`.
    fn lower_async_fn_output_type_to_future_bound(
        &mut self,
        output: &FnRetTy,
        span: Span,
        nested_impl_trait_context: ImplTraitContext,
    ) -> hir::GenericBound<'hir> {
        // Compute the `T` in `Future<Output = T>` from the return type.
        let output_ty = match output {
            FnRetTy::Ty(ty) => {
                // Not `OpaqueTyOrigin::AsyncFn`: that's only used for the
                // `impl Future` opaque type that `async fn` implicitly
                // generates.
                self.lower_ty(ty, &nested_impl_trait_context)
            }
            FnRetTy::Default(ret_ty_span) => self.arena.alloc(self.ty_tup(*ret_ty_span, &[])),
        };

        // "<Output = T>"
        let future_args = self.arena.alloc(hir::GenericArgs {
            args: &[],
            bindings: arena_vec![self; self.output_ty_binding(span, output_ty)],
            parenthesized: hir::GenericArgsParentheses::No,
            span_ext: DUMMY_SP,
        });

        hir::GenericBound::LangItemTrait(
            // ::std::future::Future<future_params>
            hir::LangItem::Future,
            self.lower_span(span),
            self.next_id(),
            future_args,
        )
    }

    #[instrument(level = "trace", skip(self))]
    fn lower_param_bound(
        &mut self,
        tpb: &GenericBound,
        itctx: &ImplTraitContext,
    ) -> hir::GenericBound<'hir> {
        match tpb {
            GenericBound::Trait(p, modifier) => hir::GenericBound::Trait(
                self.lower_poly_trait_ref(p, itctx, modifier.to_constness()),
                self.lower_trait_bound_modifier(*modifier),
            ),
            GenericBound::Outlives(lifetime) => {
                hir::GenericBound::Outlives(self.lower_lifetime(lifetime))
            }
        }
    }

    fn lower_lifetime(&mut self, l: &Lifetime) -> &'hir hir::Lifetime {
        let ident = self.lower_ident(l.ident);
        self.new_named_lifetime(l.id, l.id, ident)
    }

    #[instrument(level = "debug", skip(self))]
    fn new_named_lifetime_with_res(
        &mut self,
        id: NodeId,
        ident: Ident,
        res: LifetimeRes,
    ) -> &'hir hir::Lifetime {
        let res = match res {
            LifetimeRes::Param { param, .. } => {
                let param = self.get_remapped_def_id(param);
                hir::LifetimeName::Param(param)
            }
            LifetimeRes::Fresh { param, .. } => {
                let param = self.local_def_id(param);
                hir::LifetimeName::Param(param)
            }
            LifetimeRes::Infer => hir::LifetimeName::Infer,
            LifetimeRes::Static => hir::LifetimeName::Static,
            LifetimeRes::Error => hir::LifetimeName::Error,
            res => panic!(
                "Unexpected lifetime resolution {:?} for {:?} at {:?}",
                res, ident, ident.span
            ),
        };

        debug!(?res);
        self.arena.alloc(hir::Lifetime {
            hir_id: self.lower_node_id(id),
            ident: self.lower_ident(ident),
            res,
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn new_named_lifetime(
        &mut self,
        id: NodeId,
        new_id: NodeId,
        ident: Ident,
    ) -> &'hir hir::Lifetime {
        let res = self.resolver.get_lifetime_res(id).unwrap_or(LifetimeRes::Error);
        self.new_named_lifetime_with_res(new_id, ident, res)
    }

    fn lower_generic_params_mut<'s>(
        &'s mut self,
        params: &'s [GenericParam],
        source: hir::GenericParamSource,
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> + Captures<'a> + Captures<'s> {
        params.iter().map(move |param| self.lower_generic_param(param, source))
    }

    fn lower_generic_params(
        &mut self,
        params: &[GenericParam],
        source: hir::GenericParamSource,
    ) -> &'hir [hir::GenericParam<'hir>] {
        self.arena.alloc_from_iter(self.lower_generic_params_mut(params, source))
    }

    #[instrument(level = "trace", skip(self))]
    fn lower_generic_param(
        &mut self,
        param: &GenericParam,
        source: hir::GenericParamSource,
    ) -> hir::GenericParam<'hir> {
        let (name, kind) = self.lower_generic_param_kind(param);

        let hir_id = self.lower_node_id(param.id);
        self.lower_attrs(hir_id, &param.attrs);
        hir::GenericParam {
            hir_id,
            def_id: self.local_def_id(param.id),
            name,
            span: self.lower_span(param.span()),
            pure_wrt_drop: attr::contains_name(&param.attrs, sym::may_dangle),
            kind,
            colon_span: param.colon_span.map(|s| self.lower_span(s)),
            source,
        }
    }

    fn lower_generic_param_kind(
        &mut self,
        param: &GenericParam,
    ) -> (hir::ParamName, hir::GenericParamKind<'hir>) {
        match &param.kind {
            GenericParamKind::Lifetime => {
                // AST resolution emitted an error on those parameters, so we lower them using
                // `ParamName::Error`.
                let param_name =
                    if let Some(LifetimeRes::Error) = self.resolver.get_lifetime_res(param.id) {
                        ParamName::Error
                    } else {
                        let ident = self.lower_ident(param.ident);
                        ParamName::Plain(ident)
                    };
                let kind =
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit };

                (param_name, kind)
            }
            GenericParamKind::Type { default, .. } => {
                let kind = hir::GenericParamKind::Type {
                    default: default.as_ref().map(|x| {
                        self.lower_ty(
                            x,
                            &ImplTraitContext::Disallowed(ImplTraitPosition::GenericDefault),
                        )
                    }),
                    synthetic: false,
                };

                (hir::ParamName::Plain(self.lower_ident(param.ident)), kind)
            }
            GenericParamKind::Const { ty, kw_span: _, default } => {
                let ty = self.lower_ty(
                    &ty,
                    &ImplTraitContext::Disallowed(ImplTraitPosition::GenericDefault),
                );
                let default = default.as_ref().map(|def| self.lower_anon_const(def));
                (
                    hir::ParamName::Plain(self.lower_ident(param.ident)),
                    hir::GenericParamKind::Const { ty, default },
                )
            }
        }
    }

    fn lower_trait_ref(
        &mut self,
        constness: ast::Const,
        p: &TraitRef,
        itctx: &ImplTraitContext,
    ) -> hir::TraitRef<'hir> {
        let path = match self.lower_qpath(
            p.ref_id,
            &None,
            &p.path,
            ParamMode::Explicit,
            itctx,
            Some(constness),
        ) {
            hir::QPath::Resolved(None, path) => path,
            qpath => panic!("lower_trait_ref: unexpected QPath `{qpath:?}`"),
        };
        hir::TraitRef { path, hir_ref_id: self.lower_node_id(p.ref_id) }
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_poly_trait_ref(
        &mut self,
        p: &PolyTraitRef,
        itctx: &ImplTraitContext,
        constness: ast::Const,
    ) -> hir::PolyTraitRef<'hir> {
        let bound_generic_params =
            self.lower_lifetime_binder(p.trait_ref.ref_id, &p.bound_generic_params);
        let trait_ref = self.lower_trait_ref(constness, &p.trait_ref, itctx);
        hir::PolyTraitRef { bound_generic_params, trait_ref, span: self.lower_span(p.span) }
    }

    fn lower_mt(&mut self, mt: &MutTy, itctx: &ImplTraitContext) -> hir::MutTy<'hir> {
        hir::MutTy { ty: self.lower_ty(&mt.ty, itctx), mutbl: mt.mutbl }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn lower_param_bounds(
        &mut self,
        bounds: &[GenericBound],
        itctx: &ImplTraitContext,
    ) -> hir::GenericBounds<'hir> {
        self.arena.alloc_from_iter(self.lower_param_bounds_mut(bounds, itctx))
    }

    fn lower_param_bounds_mut<'s>(
        &'s mut self,
        bounds: &'s [GenericBound],
        itctx: &'s ImplTraitContext,
    ) -> impl Iterator<Item = hir::GenericBound<'hir>> + Captures<'s> + Captures<'a> {
        bounds.iter().map(move |bound| self.lower_param_bound(bound, itctx))
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn lower_universal_param_and_bounds(
        &mut self,
        node_id: NodeId,
        span: Span,
        ident: Ident,
        bounds: &[GenericBound],
    ) -> (hir::GenericParam<'hir>, Option<hir::WherePredicate<'hir>>, hir::TyKind<'hir>) {
        // Add a definition for the in-band `Param`.
        let def_id = self.local_def_id(node_id);
        let span = self.lower_span(span);

        // Set the name to `impl Bound1 + Bound2`.
        let param = hir::GenericParam {
            hir_id: self.lower_node_id(node_id),
            def_id,
            name: ParamName::Plain(self.lower_ident(ident)),
            pure_wrt_drop: false,
            span,
            kind: hir::GenericParamKind::Type { default: None, synthetic: true },
            colon_span: None,
            source: hir::GenericParamSource::Generics,
        };

        let preds = self.lower_generic_bound_predicate(
            ident,
            node_id,
            &GenericParamKind::Type { default: None },
            bounds,
            /* colon_span */ None,
            span,
            &ImplTraitContext::Universal,
            hir::PredicateOrigin::ImplTrait,
        );

        let hir_id = self.next_id();
        let res = Res::Def(DefKind::TyParam, def_id.to_def_id());
        let ty = hir::TyKind::Path(hir::QPath::Resolved(
            None,
            self.arena.alloc(hir::Path {
                span,
                res,
                segments:
                    arena_vec![self; hir::PathSegment::new(self.lower_ident(ident), hir_id, res)],
            }),
        ));

        (param, preds, ty)
    }

    /// Lowers a block directly to an expression, presuming that it
    /// has no attributes and is not targeted by a `break`.
    fn lower_block_expr(&mut self, b: &Block) -> hir::Expr<'hir> {
        let block = self.lower_block(b, false);
        self.expr_block(block)
    }

    fn lower_array_length(&mut self, c: &AnonConst) -> hir::ArrayLen {
        match c.value.kind {
            ExprKind::Underscore => {
                if self.tcx.features().generic_arg_infer {
                    hir::ArrayLen::Infer(self.lower_node_id(c.id), c.value.span)
                } else {
                    feature_err(
                        &self.tcx.sess.parse_sess,
                        sym::generic_arg_infer,
                        c.value.span,
                        "using `_` for array lengths is unstable",
                    )
                    .stash(c.value.span, StashKey::UnderscoreForArrayLengths);
                    hir::ArrayLen::Body(self.lower_anon_const(c))
                }
            }
            _ => hir::ArrayLen::Body(self.lower_anon_const(c)),
        }
    }

    fn lower_anon_const(&mut self, c: &AnonConst) -> hir::AnonConst {
        self.with_new_scopes(|this| hir::AnonConst {
            def_id: this.local_def_id(c.id),
            hir_id: this.lower_node_id(c.id),
            body: this.lower_const_body(c.value.span, Some(&c.value)),
        })
    }

    fn lower_unsafe_source(&mut self, u: UnsafeSource) -> hir::UnsafeSource {
        match u {
            CompilerGenerated => hir::UnsafeSource::CompilerGenerated,
            UserProvided => hir::UnsafeSource::UserProvided,
        }
    }

    fn lower_trait_bound_modifier(&mut self, f: TraitBoundModifier) -> hir::TraitBoundModifier {
        match f {
            TraitBoundModifier::None => hir::TraitBoundModifier::None,
            TraitBoundModifier::MaybeConst => hir::TraitBoundModifier::MaybeConst,

            TraitBoundModifier::Negative => {
                if self.tcx.features().negative_bounds {
                    hir::TraitBoundModifier::Negative
                } else {
                    hir::TraitBoundModifier::None
                }
            }

            // `MaybeConstMaybe` will cause an error during AST validation, but we need to pick a
            // placeholder for compilation to proceed.
            TraitBoundModifier::MaybeConstMaybe | TraitBoundModifier::Maybe => {
                hir::TraitBoundModifier::Maybe
            }
            TraitBoundModifier::MaybeConstNegative => hir::TraitBoundModifier::MaybeConst,
        }
    }

    // Helper methods for building HIR.

    fn stmt(&mut self, span: Span, kind: hir::StmtKind<'hir>) -> hir::Stmt<'hir> {
        hir::Stmt { span: self.lower_span(span), kind, hir_id: self.next_id() }
    }

    fn stmt_expr(&mut self, span: Span, expr: hir::Expr<'hir>) -> hir::Stmt<'hir> {
        self.stmt(span, hir::StmtKind::Expr(self.arena.alloc(expr)))
    }

    fn stmt_let_pat(
        &mut self,
        attrs: Option<&'hir [Attribute]>,
        span: Span,
        init: Option<&'hir hir::Expr<'hir>>,
        pat: &'hir hir::Pat<'hir>,
        source: hir::LocalSource,
    ) -> hir::Stmt<'hir> {
        let hir_id = self.next_id();
        if let Some(a) = attrs {
            debug_assert!(!a.is_empty());
            self.attrs.insert(hir_id.local_id, a);
        }
        let local = hir::Local {
            hir_id,
            init,
            pat,
            els: None,
            source,
            span: self.lower_span(span),
            ty: None,
        };
        self.stmt(span, hir::StmtKind::Local(self.arena.alloc(local)))
    }

    fn block_expr(&mut self, expr: &'hir hir::Expr<'hir>) -> &'hir hir::Block<'hir> {
        self.block_all(expr.span, &[], Some(expr))
    }

    fn block_all(
        &mut self,
        span: Span,
        stmts: &'hir [hir::Stmt<'hir>],
        expr: Option<&'hir hir::Expr<'hir>>,
    ) -> &'hir hir::Block<'hir> {
        let blk = hir::Block {
            stmts,
            expr,
            hir_id: self.next_id(),
            rules: hir::BlockCheckMode::DefaultBlock,
            span: self.lower_span(span),
            targeted_by_break: false,
        };
        self.arena.alloc(blk)
    }

    fn pat_cf_continue(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowContinue, field, None)
    }

    fn pat_cf_break(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowBreak, field, None)
    }

    fn pat_some(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::OptionSome, field, None)
    }

    fn pat_none(&mut self, span: Span) -> &'hir hir::Pat<'hir> {
        self.pat_lang_item_variant(span, hir::LangItem::OptionNone, &[], None)
    }

    fn single_pat_field(
        &mut self,
        span: Span,
        pat: &'hir hir::Pat<'hir>,
    ) -> &'hir [hir::PatField<'hir>] {
        let field = hir::PatField {
            hir_id: self.next_id(),
            ident: Ident::new(sym::integer(0), self.lower_span(span)),
            is_shorthand: false,
            pat,
            span: self.lower_span(span),
        };
        arena_vec![self; field]
    }

    fn pat_lang_item_variant(
        &mut self,
        span: Span,
        lang_item: hir::LangItem,
        fields: &'hir [hir::PatField<'hir>],
        hir_id: Option<hir::HirId>,
    ) -> &'hir hir::Pat<'hir> {
        let qpath = hir::QPath::LangItem(lang_item, self.lower_span(span), hir_id);
        self.pat(span, hir::PatKind::Struct(qpath, fields, false))
    }

    fn pat_ident(&mut self, span: Span, ident: Ident) -> (&'hir hir::Pat<'hir>, hir::HirId) {
        self.pat_ident_binding_mode(span, ident, hir::BindingAnnotation::NONE)
    }

    fn pat_ident_mut(&mut self, span: Span, ident: Ident) -> (hir::Pat<'hir>, hir::HirId) {
        self.pat_ident_binding_mode_mut(span, ident, hir::BindingAnnotation::NONE)
    }

    fn pat_ident_binding_mode(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> (&'hir hir::Pat<'hir>, hir::HirId) {
        let (pat, hir_id) = self.pat_ident_binding_mode_mut(span, ident, bm);
        (self.arena.alloc(pat), hir_id)
    }

    fn pat_ident_binding_mode_mut(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> (hir::Pat<'hir>, hir::HirId) {
        let hir_id = self.next_id();

        (
            hir::Pat {
                hir_id,
                kind: hir::PatKind::Binding(bm, hir_id, self.lower_ident(ident), None),
                span: self.lower_span(span),
                default_binding_modes: true,
            },
            hir_id,
        )
    }

    fn pat(&mut self, span: Span, kind: hir::PatKind<'hir>) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(hir::Pat {
            hir_id: self.next_id(),
            kind,
            span: self.lower_span(span),
            default_binding_modes: true,
        })
    }

    fn pat_without_dbm(&mut self, span: Span, kind: hir::PatKind<'hir>) -> hir::Pat<'hir> {
        hir::Pat {
            hir_id: self.next_id(),
            kind,
            span: self.lower_span(span),
            default_binding_modes: false,
        }
    }

    fn ty_path(
        &mut self,
        mut hir_id: hir::HirId,
        span: Span,
        qpath: hir::QPath<'hir>,
    ) -> hir::Ty<'hir> {
        let kind = match qpath {
            hir::QPath::Resolved(None, path) => {
                // Turn trait object paths into `TyKind::TraitObject` instead.
                match path.res {
                    Res::Def(DefKind::Trait | DefKind::TraitAlias, _) => {
                        let principal = hir::PolyTraitRef {
                            bound_generic_params: &[],
                            trait_ref: hir::TraitRef { path, hir_ref_id: hir_id },
                            span: self.lower_span(span),
                        };

                        // The original ID is taken by the `PolyTraitRef`,
                        // so the `Ty` itself needs a different one.
                        hir_id = self.next_id();
                        hir::TyKind::TraitObject(
                            arena_vec![self; principal],
                            self.elided_dyn_bound(span),
                            TraitObjectSyntax::None,
                        )
                    }
                    _ => hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                }
            }
            _ => hir::TyKind::Path(qpath),
        };

        hir::Ty { hir_id, kind, span: self.lower_span(span) }
    }

    /// Invoked to create the lifetime argument(s) for an elided trait object
    /// bound, like the bound in `Box<dyn Debug>`. This method is not invoked
    /// when the bound is written, even if it is written with `'_` like in
    /// `Box<dyn Debug + '_>`. In those cases, `lower_lifetime` is invoked.
    fn elided_dyn_bound(&mut self, span: Span) -> &'hir hir::Lifetime {
        let r = hir::Lifetime {
            hir_id: self.next_id(),
            ident: Ident::new(kw::Empty, self.lower_span(span)),
            res: hir::LifetimeName::ImplicitObjectLifetimeDefault,
        };
        debug!("elided_dyn_bound: r={:?}", r);
        self.arena.alloc(r)
    }
}

/// Helper struct for delayed construction of GenericArgs.
struct GenericArgsCtor<'hir> {
    args: SmallVec<[hir::GenericArg<'hir>; 4]>,
    bindings: &'hir [hir::TypeBinding<'hir>],
    parenthesized: hir::GenericArgsParentheses,
    span: Span,
}

impl<'hir> GenericArgsCtor<'hir> {
    fn push_constness(&mut self, lcx: &mut LoweringContext<'_, 'hir>, constness: ast::Const) {
        if !lcx.tcx.features().effects {
            return;
        }

        // if bound is non-const, don't add host effect param
        let ast::Const::Yes(span) = constness else { return };

        let span = lcx.lower_span(span);

        let id = lcx.next_node_id();
        let hir_id = lcx.next_id();

        let Some(host_param_id) = lcx.host_param_id else {
            lcx.tcx
                .sess
                .delay_span_bug(span, "no host param id for call in const yet no errors reported");
            return;
        };

        let body = lcx.lower_body(|lcx| {
            (&[], {
                let hir_id = lcx.next_id();
                let res = Res::Def(DefKind::ConstParam, host_param_id.to_def_id());
                let expr_kind = hir::ExprKind::Path(hir::QPath::Resolved(
                    None,
                    lcx.arena.alloc(hir::Path {
                        span,
                        res,
                        segments: arena_vec![lcx; hir::PathSegment::new(Ident {
                            name: sym::host,
                            span,
                        }, hir_id, res)],
                    }),
                ));
                lcx.expr(span, expr_kind)
            })
        });

        let attr_id = lcx.tcx.sess.parse_sess.attr_id_generator.mk_attr_id();
        let attr = lcx.arena.alloc(Attribute {
            kind: AttrKind::Normal(P(NormalAttr::from_ident(Ident::new(sym::rustc_host, span)))),
            span,
            id: attr_id,
            style: AttrStyle::Outer,
        });
        lcx.attrs.insert(hir_id.local_id, std::slice::from_ref(attr));

        let def_id =
            lcx.create_def(lcx.current_hir_id_owner.def_id, id, DefPathData::AnonConst, span);
        lcx.children.push((def_id, hir::MaybeOwner::NonOwner(hir_id)));
        self.args.push(hir::GenericArg::Const(hir::ConstArg {
            value: hir::AnonConst { def_id, hir_id, body },
            span,
        }))
    }

    fn is_empty(&self) -> bool {
        self.args.is_empty()
            && self.bindings.is_empty()
            && self.parenthesized == hir::GenericArgsParentheses::No
    }

    fn into_generic_args(self, this: &LoweringContext<'_, 'hir>) -> &'hir hir::GenericArgs<'hir> {
        let ga = hir::GenericArgs {
            args: this.arena.alloc_from_iter(self.args),
            bindings: self.bindings,
            parenthesized: self.parenthesized,
            span_ext: this.lower_span(self.span),
        };
        this.arena.alloc(ga)
    }
}
