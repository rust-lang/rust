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

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(exact_size_is_empty)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use std::sync::Arc;

use rustc_ast::node_id::NodeMap;
use rustc_ast::{self as ast, *};
use rustc_attr_parsing::{AttributeParser, OmitDoc};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::tagged_ptr::TaggedRef;
use rustc_errors::{DiagArgFromDisplay, DiagCtxtHandle, StashKey};
use rustc_hir::def::{DefKind, LifetimeRes, Namespace, PartialRes, PerNS, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, LOCAL_CRATE, LocalDefId};
use rustc_hir::{
    self as hir, ConstArg, GenericArg, HirId, ItemLocalMap, LangItem, LifetimeSource,
    LifetimeSyntax, ParamName, TraitCandidate,
};
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_macros::extension;
use rustc_middle::span_bug;
use rustc_middle::ty::{ResolverAstLowering, TyCtxt};
use rustc_session::parse::{add_feature_diagnostics, feature_err};
use rustc_span::symbol::{Ident, Symbol, kw, sym};
use rustc_span::{DUMMY_SP, DesugaringKind, Span};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;
use tracing::{debug, instrument, trace};

use crate::errors::{AssocTyParentheses, AssocTyParenthesesSub, MisplacedImplTrait};

macro_rules! arena_vec {
    ($this:expr; $($x:expr),*) => (
        $this.arena.alloc_from_iter([$($x),*])
    );
}

mod asm;
mod block;
mod delegation;
mod errors;
mod expr;
mod format;
mod index;
mod item;
mod pat;
mod path;
pub mod stability;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

struct LoweringContext<'a, 'hir> {
    tcx: TyCtxt<'hir>,
    resolver: &'a mut ResolverAstLowering,

    /// Used to allocate HIR nodes.
    arena: &'hir hir::Arena<'hir>,

    /// Bodies inside the owner being lowered.
    bodies: Vec<(hir::ItemLocalId, &'hir hir::Body<'hir>)>,
    /// `#[define_opaque]` attributes
    define_opaque: Option<&'hir [(Span, LocalDefId)]>,
    /// Attributes inside the owner being lowered.
    attrs: SortedMap<hir::ItemLocalId, &'hir [hir::Attribute]>,
    /// Collect items that were created by lowering the current owner.
    children: Vec<(LocalDefId, hir::MaybeOwner<'hir>)>,

    contract_ensures: Option<(Span, Ident, HirId)>,

    coroutine_kind: Option<hir::CoroutineKind>,

    /// When inside an `async` context, this is the `HirId` of the
    /// `task_context` local bound to the resume argument of the coroutine.
    task_context: Option<HirId>,

    /// Used to get the current `fn`'s def span to point to when using `await`
    /// outside of an `async fn`.
    current_item: Option<Span>,

    catch_scope: Option<HirId>,
    loop_scope: Option<HirId>,
    is_in_loop_condition: bool,
    is_in_dyn_type: bool,

    current_hir_id_owner: hir::OwnerId,
    item_local_id_counter: hir::ItemLocalId,
    trait_map: ItemLocalMap<Box<[TraitCandidate]>>,

    impl_trait_defs: Vec<hir::GenericParam<'hir>>,
    impl_trait_bounds: Vec<hir::WherePredicate<'hir>>,

    /// NodeIds of pattern identifiers and labelled nodes that are lowered inside the current HIR owner.
    ident_and_label_to_local_id: NodeMap<hir::ItemLocalId>,
    /// NodeIds that are lowered inside the current HIR owner. Only used for duplicate lowering check.
    #[cfg(debug_assertions)]
    node_id_to_local_id: NodeMap<hir::ItemLocalId>,

    allow_try_trait: Arc<[Symbol]>,
    allow_gen_future: Arc<[Symbol]>,
    allow_pattern_type: Arc<[Symbol]>,
    allow_async_iterator: Arc<[Symbol]>,
    allow_for_await: Arc<[Symbol]>,
    allow_async_fn_traits: Arc<[Symbol]>,

    attribute_parser: AttributeParser<'hir>,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    fn new(tcx: TyCtxt<'hir>, resolver: &'a mut ResolverAstLowering) -> Self {
        let registered_tools = tcx.registered_tools(()).iter().map(|x| x.name).collect();
        Self {
            // Pseudo-globals.
            tcx,
            resolver,
            arena: tcx.hir_arena,

            // HirId handling.
            bodies: Vec::new(),
            define_opaque: None,
            attrs: SortedMap::default(),
            children: Vec::default(),
            contract_ensures: None,
            current_hir_id_owner: hir::CRATE_OWNER_ID,
            item_local_id_counter: hir::ItemLocalId::ZERO,
            ident_and_label_to_local_id: Default::default(),
            #[cfg(debug_assertions)]
            node_id_to_local_id: Default::default(),
            trait_map: Default::default(),

            // Lowering state.
            catch_scope: None,
            loop_scope: None,
            is_in_loop_condition: false,
            is_in_dyn_type: false,
            coroutine_kind: None,
            task_context: None,
            current_item: None,
            impl_trait_defs: Vec::new(),
            impl_trait_bounds: Vec::new(),
            allow_try_trait: [sym::try_trait_v2, sym::yeet_desugar_details].into(),
            allow_pattern_type: [sym::pattern_types, sym::pattern_type_range_trait].into(),
            allow_gen_future: if tcx.features().async_fn_track_caller() {
                [sym::gen_future, sym::closure_track_caller].into()
            } else {
                [sym::gen_future].into()
            },
            allow_for_await: [sym::async_iterator].into(),
            allow_async_fn_traits: [sym::async_fn_traits].into(),
            // FIXME(gen_blocks): how does `closure_track_caller`/`async_fn_track_caller`
            // interact with `gen`/`async gen` blocks
            allow_async_iterator: [sym::gen_future, sym::async_iterator].into(),

            attribute_parser: AttributeParser::new(tcx.sess, tcx.features(), registered_tools),
        }
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'hir> {
        self.tcx.dcx()
    }
}

#[extension(trait ResolverAstLoweringExt)]
impl ResolverAstLowering {
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
    fn extra_lifetime_params(&mut self, id: NodeId) -> Vec<(Ident, NodeId, LifetimeRes)> {
        self.extra_lifetime_params_map.get(&id).cloned().unwrap_or_default()
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
    OpaqueTy { origin: hir::OpaqueTyOrigin<LocalDefId> },

    /// Treat `impl Trait` as a "trait ascription", which is like a type
    /// variable but that also enforces that a set of trait goals hold.
    ///
    /// This is useful to guide inference for unnameable types.
    InBinding,

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
    Bound,
    Generic,
    ExternFnParam,
    ClosureParam,
    PointerParam,
    FnTraitParam,
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
            ImplTraitPosition::Variable => "the type of variable bindings",
            ImplTraitPosition::Trait => "traits",
            ImplTraitPosition::Bound => "bounds",
            ImplTraitPosition::Generic => "generics",
            ImplTraitPosition::ExternFnParam => "`extern fn` parameters",
            ImplTraitPosition::ClosureParam => "closure parameters",
            ImplTraitPosition::PointerParam => "`fn` pointer parameters",
            ImplTraitPosition::FnTraitParam => "the parameters of `Fn` trait bounds",
            ImplTraitPosition::ExternFnReturn => "`extern fn` return types",
            ImplTraitPosition::ClosureReturn => "closure return types",
            ImplTraitPosition::PointerReturn => "`fn` pointer return types",
            ImplTraitPosition::FnTraitReturn => "the return type of `Fn` trait bounds",
            ImplTraitPosition::GenericDefault => "generic parameter defaults",
            ImplTraitPosition::ConstTy => "const types",
            ImplTraitPosition::StaticTy => "static types",
            ImplTraitPosition::AssocTy => "associated types",
            ImplTraitPosition::FieldTy => "field types",
            ImplTraitPosition::Cast => "cast expression types",
            ImplTraitPosition::ImplSelf => "impl headers",
            ImplTraitPosition::OffsetOf => "`offset_of!` parameters",
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

#[derive(Copy, Clone)]
enum AstOwner<'a> {
    NonOwner,
    Crate(&'a ast::Crate),
    Item(&'a ast::Item),
    AssocItem(&'a ast::AssocItem, visit::AssocCtxt),
    ForeignItem(&'a ast::ForeignItem),
}

fn index_crate<'a>(
    node_id_to_def_id: &NodeMap<LocalDefId>,
    krate: &'a Crate,
) -> IndexVec<LocalDefId, AstOwner<'a>> {
    let mut indexer = Indexer { node_id_to_def_id, index: IndexVec::new() };
    *indexer.index.ensure_contains_elem(CRATE_DEF_ID, || AstOwner::NonOwner) =
        AstOwner::Crate(krate);
    visit::walk_crate(&mut indexer, krate);
    return indexer.index;

    struct Indexer<'s, 'a> {
        node_id_to_def_id: &'s NodeMap<LocalDefId>,
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
            visit::walk_item(self, item);
        }
    }
}

/// Compute the hash for the HIR of the full crate.
/// This hash will then be part of the crate_hash which is stored in the metadata.
fn compute_hir_hash(
    tcx: TyCtxt<'_>,
    owners: &IndexSlice<LocalDefId, hir::MaybeOwner<'_>>,
) -> Fingerprint {
    let mut hir_body_nodes: Vec<_> = owners
        .iter_enumerated()
        .filter_map(|(def_id, info)| {
            let info = info.as_owner()?;
            let def_path_hash = tcx.hir_def_path_hash(def_id);
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
    tcx.ensure_done().output_filenames(());
    tcx.ensure_done().early_lint_checks(());
    tcx.ensure_done().debugger_visualizers(LOCAL_CRATE);
    tcx.ensure_done().get_lang_items(());
    let (mut resolver, krate) = tcx.resolver_for_lowering().steal();

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

    // Don't hash unless necessary, because it's expensive.
    let opt_hir_hash =
        if tcx.needs_crate_hash() { Some(compute_hir_hash(tcx, &owners)) } else { None };
    hir::Crate { owners, opt_hir_hash }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum ParamMode {
    /// Any path in a type context.
    Explicit,
    /// The `module::Type` in `module::Type::method` in an expression.
    Optional,
}

#[derive(Copy, Clone, Debug)]
enum AllowReturnTypeNotation {
    /// Only in types, since RTN is denied later during HIR lowering.
    Yes,
    /// All other positions (path expr, method, use tree).
    No,
}

enum GenericArgsMode {
    /// Allow paren sugar, don't allow RTN.
    ParenSugar,
    /// Allow RTN, don't allow paren sugar.
    ReturnTypeNotation,
    // Error if parenthesized generics or RTN are encountered.
    Err,
    /// Silence errors when lowering generics. Only used with `Res::Err`.
    Silence,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        name: Option<Symbol>,
        def_kind: DefKind,
        span: Span,
    ) -> LocalDefId {
        debug_assert_ne!(node_id, ast::DUMMY_NODE_ID);
        assert!(
            self.opt_local_def_id(node_id).is_none(),
            "adding a def'n for node-id {:?} and def kind {:?} but a previous def'n exists: {:?}",
            node_id,
            def_kind,
            self.tcx.hir_def_key(self.local_def_id(node_id)),
        );

        let def_id = self.tcx.at(span).create_def(parent, name, def_kind).def_id();

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
    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.resolver.node_id_to_def_id.get(&node).copied()
    }

    fn local_def_id(&self, node: NodeId) -> LocalDefId {
        self.opt_local_def_id(node).unwrap_or_else(|| panic!("no entry for node id: `{node:?}`"))
    }

    /// Given the id of an owner node in the AST, returns the corresponding `OwnerId`.
    fn owner_id(&self, node: NodeId) -> hir::OwnerId {
        hir::OwnerId { def_id: self.local_def_id(node) }
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
        let owner_id = self.owner_id(owner);

        let current_attrs = std::mem::take(&mut self.attrs);
        let current_bodies = std::mem::take(&mut self.bodies);
        let current_define_opaque = std::mem::take(&mut self.define_opaque);
        let current_ident_and_label_to_local_id =
            std::mem::take(&mut self.ident_and_label_to_local_id);

        #[cfg(debug_assertions)]
        let current_node_id_to_local_id = std::mem::take(&mut self.node_id_to_local_id);
        let current_trait_map = std::mem::take(&mut self.trait_map);
        let current_owner = std::mem::replace(&mut self.current_hir_id_owner, owner_id);
        let current_local_counter =
            std::mem::replace(&mut self.item_local_id_counter, hir::ItemLocalId::new(1));
        let current_impl_trait_defs = std::mem::take(&mut self.impl_trait_defs);
        let current_impl_trait_bounds = std::mem::take(&mut self.impl_trait_bounds);

        // Do not reset `next_node_id` and `node_id_to_def_id`:
        // we want `f` to be able to refer to the `LocalDefId`s that the caller created.
        // and the caller to refer to some of the subdefinitions' nodes' `LocalDefId`s.

        // Always allocate the first `HirId` for the owner itself.
        #[cfg(debug_assertions)]
        {
            let _old = self.node_id_to_local_id.insert(owner, hir::ItemLocalId::ZERO);
            debug_assert_eq!(_old, None);
        }

        let item = f(self);
        debug_assert_eq!(owner_id, item.def_id());
        // `f` should have consumed all the elements in these vectors when constructing `item`.
        debug_assert!(self.impl_trait_defs.is_empty());
        debug_assert!(self.impl_trait_bounds.is_empty());
        let info = self.make_owner_info(item);

        self.attrs = current_attrs;
        self.bodies = current_bodies;
        self.define_opaque = current_define_opaque;
        self.ident_and_label_to_local_id = current_ident_and_label_to_local_id;

        #[cfg(debug_assertions)]
        {
            self.node_id_to_local_id = current_node_id_to_local_id;
        }
        self.trait_map = current_trait_map;
        self.current_hir_id_owner = current_owner;
        self.item_local_id_counter = current_local_counter;
        self.impl_trait_defs = current_impl_trait_defs;
        self.impl_trait_bounds = current_impl_trait_bounds;

        debug_assert!(!self.children.iter().any(|(id, _)| id == &owner_id.def_id));
        self.children.push((owner_id.def_id, hir::MaybeOwner::Owner(info)));
    }

    fn make_owner_info(&mut self, node: hir::OwnerNode<'hir>) -> &'hir hir::OwnerInfo<'hir> {
        let attrs = std::mem::take(&mut self.attrs);
        let mut bodies = std::mem::take(&mut self.bodies);
        let define_opaque = std::mem::take(&mut self.define_opaque);
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
        let (opt_hash_including_bodies, attrs_hash) =
            self.tcx.hash_owner_nodes(node, &bodies, &attrs, define_opaque);
        let num_nodes = self.item_local_id_counter.as_usize();
        let (nodes, parenting) = index::index_hir(self.tcx, node, &bodies, num_nodes);
        let nodes = hir::OwnerNodes { opt_hash_including_bodies, nodes, bodies };
        let attrs = hir::AttributeMap { map: attrs, opt_hash: attrs_hash, define_opaque };

        self.arena.alloc(hir::OwnerInfo { nodes, parenting, attrs, trait_map })
    }

    /// This method allocates a new `HirId` for the given `NodeId`.
    /// Take care not to call this method if the resulting `HirId` is then not
    /// actually used in the HIR, as that would trigger an assertion in the
    /// `HirIdValidator` later on, which makes sure that all `NodeId`s got mapped
    /// properly. Calling the method twice with the same `NodeId` is also forbidden.
    #[instrument(level = "debug", skip(self), ret)]
    fn lower_node_id(&mut self, ast_node_id: NodeId) -> HirId {
        assert_ne!(ast_node_id, DUMMY_NODE_ID);

        let owner = self.current_hir_id_owner;
        let local_id = self.item_local_id_counter;
        assert_ne!(local_id, hir::ItemLocalId::ZERO);
        self.item_local_id_counter.increment_by(1);
        let hir_id = HirId { owner, local_id };

        if let Some(def_id) = self.opt_local_def_id(ast_node_id) {
            self.children.push((def_id, hir::MaybeOwner::NonOwner(hir_id)));
        }

        if let Some(traits) = self.resolver.trait_map.remove(&ast_node_id) {
            self.trait_map.insert(hir_id.local_id, traits.into_boxed_slice());
        }

        // Check whether the same `NodeId` is lowered more than once.
        #[cfg(debug_assertions)]
        {
            let old = self.node_id_to_local_id.insert(ast_node_id, local_id);
            assert_eq!(old, None);
        }

        hir_id
    }

    /// Generate a new `HirId` without a backing `NodeId`.
    #[instrument(level = "debug", skip(self), ret)]
    fn next_id(&mut self) -> HirId {
        let owner = self.current_hir_id_owner;
        let local_id = self.item_local_id_counter;
        assert_ne!(local_id, hir::ItemLocalId::ZERO);
        self.item_local_id_counter.increment_by(1);
        HirId { owner, local_id }
    }

    #[instrument(level = "trace", skip(self))]
    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        let res: Result<Res, ()> = res.apply_id(|id| {
            let owner = self.current_hir_id_owner;
            let local_id = self.ident_and_label_to_local_id.get(&id).copied().ok_or(())?;
            Ok(HirId { owner, local_id })
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

    fn lower_import_res(&mut self, id: NodeId, span: Span) -> SmallVec<[Res; 3]> {
        let res = self.resolver.get_import_res(id).present_items();
        let res: SmallVec<_> = res.map(|res| self.lower_res(res)).collect();
        if res.is_empty() {
            self.dcx().span_delayed_bug(span, "no resolution for an import");
            return smallvec![Res::Err];
        }
        res
    }

    fn make_lang_item_qpath(
        &mut self,
        lang_item: hir::LangItem,
        span: Span,
        args: Option<&'hir hir::GenericArgs<'hir>>,
    ) -> hir::QPath<'hir> {
        hir::QPath::Resolved(None, self.make_lang_item_path(lang_item, span, args))
    }

    fn make_lang_item_path(
        &mut self,
        lang_item: hir::LangItem,
        span: Span,
        args: Option<&'hir hir::GenericArgs<'hir>>,
    ) -> &'hir hir::Path<'hir> {
        let def_id = self.tcx.require_lang_item(lang_item, Some(span));
        let def_kind = self.tcx.def_kind(def_id);
        let res = Res::Def(def_kind, def_id);
        self.arena.alloc(hir::Path {
            span,
            res,
            segments: self.arena.alloc_from_iter([hir::PathSegment {
                ident: Ident::new(lang_item.name(), span),
                hir_id: self.next_id(),
                res,
                args,
                infer_args: args.is_none(),
            }]),
        })
    }

    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    fn mark_span_with_reason(
        &self,
        reason: DesugaringKind,
        span: Span,
        allow_internal_unstable: Option<Arc<[Symbol]>>,
    ) -> Span {
        self.tcx.with_stable_hashing_context(|hcx| {
            span.mark_with_reason(allow_internal_unstable, reason, span.edition(), hcx)
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
            LifetimeRes::Fresh { param, kind, .. } => {
                // Late resolution delegates to us the creation of the `LocalDefId`.
                let _def_id = self.create_def(
                    self.current_hir_id_owner.def_id,
                    param,
                    Some(kw::UnderscoreLifetime),
                    DefKind::LifetimeParam,
                    ident.span,
                );
                debug!(?_def_id);

                (hir::ParamName::Fresh, hir::LifetimeParamKind::Elided(kind))
            }
            LifetimeRes::Static { .. } | LifetimeRes::Error => return None,
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
        let extra_lifetimes = self.resolver.extra_lifetime_params(binder);
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

    fn with_new_scopes<T>(&mut self, scope_span: Span, f: impl FnOnce(&mut Self) -> T) -> T {
        let current_item = self.current_item;
        self.current_item = Some(scope_span);

        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let old_contract = self.contract_ensures.take();

        let catch_scope = self.catch_scope.take();
        let loop_scope = self.loop_scope.take();
        let ret = f(self);
        self.catch_scope = catch_scope;
        self.loop_scope = loop_scope;

        self.contract_ensures = old_contract;

        self.is_in_loop_condition = was_in_loop_condition;

        self.current_item = current_item;

        ret
    }

    fn lower_attrs(
        &mut self,
        id: HirId,
        attrs: &[Attribute],
        target_span: Span,
    ) -> &'hir [hir::Attribute] {
        if attrs.is_empty() {
            &[]
        } else {
            let lowered_attrs = self.lower_attrs_vec(attrs, self.lower_span(target_span));

            debug_assert_eq!(id.owner, self.current_hir_id_owner);
            let ret = self.arena.alloc_from_iter(lowered_attrs);

            // this is possible if an item contained syntactical attribute,
            // but none of them parse succesfully or all of them were ignored
            // for not being built-in attributes at all. They could be remaining
            // unexpanded attributes used as markers in proc-macro derives for example.
            // This will have emitted some diagnostics for the misparse, but will then
            // not emit the attribute making the list empty.
            if ret.is_empty() {
                &[]
            } else {
                self.attrs.insert(id.local_id, ret);
                ret
            }
        }
    }

    fn lower_attrs_vec(&self, attrs: &[Attribute], target_span: Span) -> Vec<hir::Attribute> {
        self.attribute_parser
            .parse_attribute_list(attrs, target_span, OmitDoc::Lower, |s| self.lower_span(s))
    }

    fn alias_attrs(&mut self, id: HirId, target_id: HirId) {
        debug_assert_eq!(id.owner, self.current_hir_id_owner);
        debug_assert_eq!(target_id.owner, self.current_hir_id_owner);
        if let Some(&a) = self.attrs.get(&target_id.local_id) {
            debug_assert!(!a.is_empty());
            self.attrs.insert(id.local_id, a);
        }
    }

    fn lower_delim_args(&self, args: &DelimArgs) -> DelimArgs {
        DelimArgs { dspan: args.dspan, delim: args.delim, tokens: args.tokens.clone() }
    }

    /// Lower an associated item constraint.
    #[instrument(level = "debug", skip_all)]
    fn lower_assoc_item_constraint(
        &mut self,
        constraint: &AssocItemConstraint,
        itctx: ImplTraitContext,
    ) -> hir::AssocItemConstraint<'hir> {
        debug!(?constraint, ?itctx);
        // Lower the generic arguments for the associated item.
        let gen_args = if let Some(gen_args) = &constraint.gen_args {
            let gen_args_ctor = match gen_args {
                GenericArgs::AngleBracketed(data) => {
                    self.lower_angle_bracketed_parameter_data(data, ParamMode::Explicit, itctx).0
                }
                GenericArgs::Parenthesized(data) => {
                    if let Some(first_char) = constraint.ident.as_str().chars().next()
                        && first_char.is_ascii_lowercase()
                    {
                        let err = match (&data.inputs[..], &data.output) {
                            ([_, ..], FnRetTy::Default(_)) => {
                                errors::BadReturnTypeNotation::Inputs { span: data.inputs_span }
                            }
                            ([], FnRetTy::Default(_)) => {
                                errors::BadReturnTypeNotation::NeedsDots { span: data.inputs_span }
                            }
                            // The case `T: Trait<method(..) -> Ret>` is handled in the parser.
                            (_, FnRetTy::Ty(ty)) => {
                                let span = data.inputs_span.shrink_to_hi().to(ty.span);
                                errors::BadReturnTypeNotation::Output {
                                    span,
                                    suggestion: errors::RTNSuggestion {
                                        output: span,
                                        input: data.inputs_span,
                                    },
                                }
                            }
                        };
                        let mut err = self.dcx().create_err(err);
                        if !self.tcx.features().return_type_notation()
                            && self.tcx.sess.is_nightly_build()
                        {
                            add_feature_diagnostics(
                                &mut err,
                                &self.tcx.sess,
                                sym::return_type_notation,
                            );
                        }
                        err.emit();
                        GenericArgsCtor {
                            args: Default::default(),
                            constraints: &[],
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
                GenericArgs::ParenthesizedElided(span) => GenericArgsCtor {
                    args: Default::default(),
                    constraints: &[],
                    parenthesized: hir::GenericArgsParentheses::ReturnTypeNotation,
                    span: *span,
                },
            };
            gen_args_ctor.into_generic_args(self)
        } else {
            self.arena.alloc(hir::GenericArgs::none())
        };
        let kind = match &constraint.kind {
            AssocItemConstraintKind::Equality { term } => {
                let term = match term {
                    Term::Ty(ty) => self.lower_ty(ty, itctx).into(),
                    Term::Const(c) => self.lower_anon_const_to_const_arg(c).into(),
                };
                hir::AssocItemConstraintKind::Equality { term }
            }
            AssocItemConstraintKind::Bound { bounds } => {
                // Disallow ATB in dyn types
                if self.is_in_dyn_type {
                    let suggestion = match itctx {
                        ImplTraitContext::OpaqueTy { .. } | ImplTraitContext::Universal => {
                            let bound_end_span = constraint
                                .gen_args
                                .as_ref()
                                .map_or(constraint.ident.span, |args| args.span());
                            if bound_end_span.eq_ctxt(constraint.span) {
                                Some(self.tcx.sess.source_map().next_point(bound_end_span))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    let guar = self.dcx().emit_err(errors::MisplacedAssocTyBinding {
                        span: constraint.span,
                        suggestion,
                    });
                    let err_ty =
                        &*self.arena.alloc(self.ty(constraint.span, hir::TyKind::Err(guar)));
                    hir::AssocItemConstraintKind::Equality { term: err_ty.into() }
                } else {
                    // Desugar `AssocTy: Bounds` into an assoc type binding where the
                    // later desugars into a trait predicate.
                    let bounds = self.lower_param_bounds(bounds, itctx);

                    hir::AssocItemConstraintKind::Bound { bounds }
                }
            }
        };

        hir::AssocItemConstraint {
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
        self.dcx().emit_err(AssocTyParentheses { span: data.span, sub });
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_generic_arg(
        &mut self,
        arg: &ast::GenericArg,
        itctx: ImplTraitContext,
    ) -> hir::GenericArg<'hir> {
        match arg {
            ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.lower_lifetime(
                lt,
                LifetimeSource::Path { with_angle_brackets: true },
                lt.ident.into(),
            )),
            ast::GenericArg::Type(ty) => {
                // We cannot just match on `TyKind::Infer` as `(_)` is represented as
                // `TyKind::Paren(TyKind::Infer)` and should also be lowered to `GenericArg::Infer`
                if ty.is_maybe_parenthesised_infer() {
                    return GenericArg::Infer(hir::InferArg {
                        hir_id: self.lower_node_id(ty.id),
                        span: self.lower_span(ty.span),
                    });
                }

                match &ty.kind {
                    // We parse const arguments as path types as we cannot distinguish them during
                    // parsing. We try to resolve that ambiguity by attempting resolution in both the
                    // type and value namespaces. If we resolved the path in the value namespace, we
                    // transform it into a generic const argument.
                    //
                    // FIXME: Should we be handling `(PATH_TO_CONST)`?
                    TyKind::Path(None, path) => {
                        if let Some(res) = self
                            .resolver
                            .get_partial_res(ty.id)
                            .and_then(|partial_res| partial_res.full_res())
                        {
                            if !res.matches_ns(Namespace::TypeNS)
                                && path.is_potential_trivial_const_arg(false)
                            {
                                debug!(
                                    "lower_generic_arg: Lowering type argument as const argument: {:?}",
                                    ty,
                                );

                                let ct =
                                    self.lower_const_path_to_const_arg(path, res, ty.id, ty.span);
                                return GenericArg::Const(ct.try_as_ambig_ct().unwrap());
                            }
                        }
                    }
                    _ => {}
                }
                GenericArg::Type(self.lower_ty(ty, itctx).try_as_ambig_ty().unwrap())
            }
            ast::GenericArg::Const(ct) => {
                GenericArg::Const(self.lower_anon_const_to_const_arg(ct).try_as_ambig_ct().unwrap())
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_ty(&mut self, t: &Ty, itctx: ImplTraitContext) -> &'hir hir::Ty<'hir> {
        self.arena.alloc(self.lower_ty_direct(t, itctx))
    }

    fn lower_path_ty(
        &mut self,
        t: &Ty,
        qself: &Option<ptr::P<QSelf>>,
        path: &Path,
        param_mode: ParamMode,
        itctx: ImplTraitContext,
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
                        modifiers: TraitBoundModifiers::NONE,
                        trait_ref: TraitRef { path: path.clone(), ref_id: t.id },
                        span: t.span,
                    },
                    itctx,
                );
                let bounds = this.arena.alloc_from_iter([bound]);
                let lifetime_bound = this.elided_dyn_bound(t.span);
                (bounds, lifetime_bound)
            });
            let kind = hir::TyKind::TraitObject(
                bounds,
                TaggedRef::new(lifetime_bound, TraitObjectSyntax::None),
            );
            return hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.next_id() };
        }

        let id = self.lower_node_id(t.id);
        let qpath = self.lower_qpath(
            t.id,
            qself,
            path,
            param_mode,
            AllowReturnTypeNotation::Yes,
            itctx,
            None,
        );
        self.ty_path(id, t.span, qpath)
    }

    fn ty(&mut self, span: Span, kind: hir::TyKind<'hir>) -> hir::Ty<'hir> {
        hir::Ty { hir_id: self.next_id(), kind, span: self.lower_span(span) }
    }

    fn ty_tup(&mut self, span: Span, tys: &'hir [hir::Ty<'hir>]) -> hir::Ty<'hir> {
        self.ty(span, hir::TyKind::Tup(tys))
    }

    fn lower_ty_direct(&mut self, t: &Ty, itctx: ImplTraitContext) -> hir::Ty<'hir> {
        let kind = match &t.kind {
            TyKind::Infer => hir::TyKind::Infer(()),
            TyKind::Err(guar) => hir::TyKind::Err(*guar),
            TyKind::Slice(ty) => hir::TyKind::Slice(self.lower_ty(ty, itctx)),
            TyKind::Ptr(mt) => hir::TyKind::Ptr(self.lower_mt(mt, itctx)),
            TyKind::Ref(region, mt) => {
                let lifetime = self.lower_ty_direct_lifetime(t, *region);
                hir::TyKind::Ref(lifetime, self.lower_mt(mt, itctx))
            }
            TyKind::PinnedRef(region, mt) => {
                let lifetime = self.lower_ty_direct_lifetime(t, *region);
                let kind = hir::TyKind::Ref(lifetime, self.lower_mt(mt, itctx));
                let span = self.lower_span(t.span);
                let arg = hir::Ty { kind, span, hir_id: self.next_id() };
                let args = self.arena.alloc(hir::GenericArgs {
                    args: self.arena.alloc([hir::GenericArg::Type(self.arena.alloc(arg))]),
                    constraints: &[],
                    parenthesized: hir::GenericArgsParentheses::No,
                    span_ext: span,
                });
                let path = self.make_lang_item_qpath(LangItem::Pin, span, Some(args));
                hir::TyKind::Path(path)
            }
            TyKind::BareFn(f) => {
                let generic_params = self.lower_lifetime_binder(t.id, &f.generic_params);
                hir::TyKind::BareFn(self.arena.alloc(hir::BareFnTy {
                    generic_params,
                    safety: self.lower_safety(f.safety, hir::Safety::Safe),
                    abi: self.lower_extern(f.ext),
                    decl: self.lower_fn_decl(&f.decl, t.id, t.span, FnDeclKind::Pointer, None),
                    param_idents: self.lower_fn_params_to_idents(&f.decl),
                }))
            }
            TyKind::UnsafeBinder(f) => {
                let generic_params = self.lower_lifetime_binder(t.id, &f.generic_params);
                hir::TyKind::UnsafeBinder(self.arena.alloc(hir::UnsafeBinderTy {
                    generic_params,
                    inner_ty: self.lower_ty(&f.inner_ty, itctx),
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
            TyKind::Array(ty, length) => hir::TyKind::Array(
                self.lower_ty(ty, itctx),
                self.lower_array_length_to_const_arg(length),
            ),
            TyKind::Typeof(expr) => hir::TyKind::Typeof(self.lower_anon_const_to_anon_const(expr)),
            TyKind::TraitObject(bounds, kind) => {
                let mut lifetime_bound = None;
                let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                    let bounds =
                        this.arena.alloc_from_iter(bounds.iter().filter_map(|bound| match bound {
                            // We can safely ignore constness here since AST validation
                            // takes care of rejecting invalid modifier combinations and
                            // const trait bounds in trait object types.
                            GenericBound::Trait(ty) => {
                                let trait_ref = this.lower_poly_trait_ref(ty, itctx);
                                Some(trait_ref)
                            }
                            GenericBound::Outlives(lifetime) => {
                                if lifetime_bound.is_none() {
                                    lifetime_bound = Some(this.lower_lifetime(
                                        lifetime,
                                        LifetimeSource::Other,
                                        lifetime.ident.into(),
                                    ));
                                }
                                None
                            }
                            // Ignore `use` syntax since that is not valid in objects.
                            GenericBound::Use(_, span) => {
                                this.dcx()
                                    .span_delayed_bug(*span, "use<> not allowed in dyn types");
                                None
                            }
                        }));
                    let lifetime_bound =
                        lifetime_bound.unwrap_or_else(|| this.elided_dyn_bound(t.span));
                    (bounds, lifetime_bound)
                });
                hir::TyKind::TraitObject(bounds, TaggedRef::new(lifetime_bound, *kind))
            }
            TyKind::ImplTrait(def_node_id, bounds) => {
                let span = t.span;
                match itctx {
                    ImplTraitContext::OpaqueTy { origin } => {
                        self.lower_opaque_impl_trait(span, origin, *def_node_id, bounds, itctx)
                    }
                    ImplTraitContext::Universal => {
                        if let Some(span) = bounds.iter().find_map(|bound| match *bound {
                            ast::GenericBound::Use(_, span) => Some(span),
                            _ => None,
                        }) {
                            self.tcx.dcx().emit_err(errors::NoPreciseCapturesOnApit { span });
                        }

                        let def_id = self.local_def_id(*def_node_id);
                        let name = self.tcx.item_name(def_id.to_def_id());
                        let ident = Ident::new(name, span);
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
                    ImplTraitContext::InBinding => {
                        hir::TyKind::TraitAscription(self.lower_param_bounds(bounds, itctx))
                    }
                    ImplTraitContext::FeatureGated(position, feature) => {
                        let guar = self
                            .tcx
                            .sess
                            .create_feature_err(
                                MisplacedImplTrait {
                                    span: t.span,
                                    position: DiagArgFromDisplay(&position),
                                },
                                feature,
                            )
                            .emit();
                        hir::TyKind::Err(guar)
                    }
                    ImplTraitContext::Disallowed(position) => {
                        let guar = self.dcx().emit_err(MisplacedImplTrait {
                            span: t.span,
                            position: DiagArgFromDisplay(&position),
                        });
                        hir::TyKind::Err(guar)
                    }
                }
            }
            TyKind::Pat(ty, pat) => {
                hir::TyKind::Pat(self.lower_ty(ty, itctx), self.lower_ty_pat(pat, ty.span))
            }
            TyKind::MacCall(_) => {
                span_bug!(t.span, "`TyKind::MacCall` should have been expanded by now")
            }
            TyKind::CVarArgs => {
                let guar = self.dcx().span_delayed_bug(
                    t.span,
                    "`TyKind::CVarArgs` should have been handled elsewhere",
                );
                hir::TyKind::Err(guar)
            }
            TyKind::Dummy => panic!("`TyKind::Dummy` should never be lowered"),
        };

        hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.lower_node_id(t.id) }
    }

    fn lower_ty_direct_lifetime(
        &mut self,
        t: &Ty,
        region: Option<Lifetime>,
    ) -> &'hir hir::Lifetime {
        let (region, syntax) = match region {
            Some(region) => (region, region.ident.into()),

            None => {
                let id = if let Some(LifetimeRes::ElidedAnchor { start, end }) =
                    self.resolver.get_lifetime_res(t.id)
                {
                    debug_assert_eq!(start.plus(1), end);
                    start
                } else {
                    self.next_node_id()
                };
                let span = self.tcx.sess.source_map().start_point(t.span).shrink_to_hi();
                let region = Lifetime { ident: Ident::new(kw::UnderscoreLifetime, span), id };
                (region, LifetimeSyntax::Hidden)
            }
        };
        self.lower_lifetime(&region, LifetimeSource::Reference, syntax)
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
        origin: hir::OpaqueTyOrigin<LocalDefId>,
        opaque_ty_node_id: NodeId,
        bounds: &GenericBounds,
        itctx: ImplTraitContext,
    ) -> hir::TyKind<'hir> {
        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::OpaqueTy, span, None);

        self.lower_opaque_inner(opaque_ty_node_id, origin, opaque_ty_span, |this| {
            this.lower_param_bounds(bounds, itctx)
        })
    }

    fn lower_opaque_inner(
        &mut self,
        opaque_ty_node_id: NodeId,
        origin: hir::OpaqueTyOrigin<LocalDefId>,
        opaque_ty_span: Span,
        lower_item_bounds: impl FnOnce(&mut Self) -> &'hir [hir::GenericBound<'hir>],
    ) -> hir::TyKind<'hir> {
        let opaque_ty_def_id = self.local_def_id(opaque_ty_node_id);
        let opaque_ty_hir_id = self.lower_node_id(opaque_ty_node_id);
        debug!(?opaque_ty_def_id, ?opaque_ty_hir_id);

        let bounds = lower_item_bounds(self);
        let opaque_ty_def = hir::OpaqueTy {
            hir_id: opaque_ty_hir_id,
            def_id: opaque_ty_def_id,
            bounds,
            origin,
            span: self.lower_span(opaque_ty_span),
        };
        let opaque_ty_def = self.arena.alloc(opaque_ty_def);

        hir::TyKind::OpaqueDef(opaque_ty_def)
    }

    fn lower_precise_capturing_args(
        &mut self,
        precise_capturing_args: &[PreciseCapturingArg],
    ) -> &'hir [hir::PreciseCapturingArg<'hir>] {
        self.arena.alloc_from_iter(precise_capturing_args.iter().map(|arg| match arg {
            PreciseCapturingArg::Lifetime(lt) => hir::PreciseCapturingArg::Lifetime(
                self.lower_lifetime(lt, LifetimeSource::PreciseCapturing, lt.ident.into()),
            ),
            PreciseCapturingArg::Arg(path, id) => {
                let [segment] = path.segments.as_slice() else {
                    panic!();
                };
                let res = self.resolver.get_partial_res(*id).map_or(Res::Err, |partial_res| {
                    partial_res.full_res().expect("no partial res expected for precise capture arg")
                });
                hir::PreciseCapturingArg::Param(hir::PreciseCapturingNonLifetimeArg {
                    hir_id: self.lower_node_id(*id),
                    ident: self.lower_ident(segment.ident),
                    res: self.lower_res(res),
                })
            }
        }))
    }

    fn lower_fn_params_to_idents(&mut self, decl: &FnDecl) -> &'hir [Option<Ident>] {
        self.arena.alloc_from_iter(decl.inputs.iter().map(|param| match param.pat.kind {
            PatKind::Missing => None,
            PatKind::Ident(_, ident, _) => Some(self.lower_ident(ident)),
            PatKind::Wild => Some(Ident::new(kw::Underscore, self.lower_span(param.pat.span))),
            _ => {
                self.dcx().span_delayed_bug(
                    param.pat.span,
                    "non-missing/ident/wild param pat must trigger an error",
                );
                None
            }
        }))
    }

    /// Lowers a function declaration.
    ///
    /// `decl`: the unlowered (AST) function declaration.
    ///
    /// `fn_node_id`: `impl Trait` arguments are lowered into generic parameters on the given
    /// `NodeId`.
    ///
    /// `transform_return_type`: if `Some`, applies some conversion to the return type, such as is
    /// needed for `async fn` and `gen fn`. See [`CoroutineKind`] for more details.
    #[instrument(level = "debug", skip(self))]
    fn lower_fn_decl(
        &mut self,
        decl: &FnDecl,
        fn_node_id: NodeId,
        fn_span: Span,
        kind: FnDeclKind,
        coro: Option<CoroutineKind>,
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
            let itctx = match kind {
                FnDeclKind::Fn | FnDeclKind::Inherent | FnDeclKind::Impl | FnDeclKind::Trait => {
                    ImplTraitContext::Universal
                }
                FnDeclKind::ExternFn => {
                    ImplTraitContext::Disallowed(ImplTraitPosition::ExternFnParam)
                }
                FnDeclKind::Closure => {
                    ImplTraitContext::Disallowed(ImplTraitPosition::ClosureParam)
                }
                FnDeclKind::Pointer => {
                    ImplTraitContext::Disallowed(ImplTraitPosition::PointerParam)
                }
            };
            self.lower_ty_direct(&param.ty, itctx)
        }));

        let output = match coro {
            Some(coro) => {
                let fn_def_id = self.local_def_id(fn_node_id);
                self.lower_coroutine_fn_ret_ty(&decl.output, fn_def_id, coro, kind, fn_span)
            }
            None => match &decl.output {
                FnRetTy::Ty(ty) => {
                    let itctx = match kind {
                        FnDeclKind::Fn | FnDeclKind::Inherent => ImplTraitContext::OpaqueTy {
                            origin: hir::OpaqueTyOrigin::FnReturn {
                                parent: self.local_def_id(fn_node_id),
                                in_trait_or_impl: None,
                            },
                        },
                        FnDeclKind::Trait => ImplTraitContext::OpaqueTy {
                            origin: hir::OpaqueTyOrigin::FnReturn {
                                parent: self.local_def_id(fn_node_id),
                                in_trait_or_impl: Some(hir::RpitContext::Trait),
                            },
                        },
                        FnDeclKind::Impl => ImplTraitContext::OpaqueTy {
                            origin: hir::OpaqueTyOrigin::FnReturn {
                                parent: self.local_def_id(fn_node_id),
                                in_trait_or_impl: Some(hir::RpitContext::TraitImpl),
                            },
                        },
                        FnDeclKind::ExternFn => {
                            ImplTraitContext::Disallowed(ImplTraitPosition::ExternFnReturn)
                        }
                        FnDeclKind::Closure => {
                            ImplTraitContext::Disallowed(ImplTraitPosition::ClosureReturn)
                        }
                        FnDeclKind::Pointer => {
                            ImplTraitContext::Disallowed(ImplTraitPosition::PointerReturn)
                        }
                    };
                    hir::FnRetTy::Return(self.lower_ty(ty, itctx))
                }
                FnRetTy::Default(span) => hir::FnRetTy::DefaultReturn(self.lower_span(*span)),
            },
        };

        self.arena.alloc(hir::FnDecl {
            inputs,
            output,
            c_variadic,
            lifetime_elision_allowed: self.resolver.lifetime_elision_allowed.contains(&fn_node_id),
            implicit_self: decl.inputs.get(0).map_or(hir::ImplicitSelfKind::None, |arg| {
                let is_mutable_pat = matches!(
                    arg.pat.kind,
                    PatKind::Ident(hir::BindingMode(_, Mutability::Mut), ..)
                );

                match &arg.ty.kind {
                    TyKind::ImplicitSelf if is_mutable_pat => hir::ImplicitSelfKind::Mut,
                    TyKind::ImplicitSelf => hir::ImplicitSelfKind::Imm,
                    // Given we are only considering `ImplicitSelf` types, we needn't consider
                    // the case where we have a mutable pattern to a reference as that would
                    // no longer be an `ImplicitSelf`.
                    TyKind::Ref(_, mt) | TyKind::PinnedRef(_, mt)
                        if mt.ty.kind.is_implicit_self() =>
                    {
                        match mt.mutbl {
                            hir::Mutability::Not => hir::ImplicitSelfKind::RefImm,
                            hir::Mutability::Mut => hir::ImplicitSelfKind::RefMut,
                        }
                    }
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
    fn lower_coroutine_fn_ret_ty(
        &mut self,
        output: &FnRetTy,
        fn_def_id: LocalDefId,
        coro: CoroutineKind,
        fn_kind: FnDeclKind,
        fn_span: Span,
    ) -> hir::FnRetTy<'hir> {
        let span = self.lower_span(fn_span);

        let (opaque_ty_node_id, allowed_features) = match coro {
            CoroutineKind::Async { return_impl_trait_id, .. } => (return_impl_trait_id, None),
            CoroutineKind::Gen { return_impl_trait_id, .. } => (return_impl_trait_id, None),
            CoroutineKind::AsyncGen { return_impl_trait_id, .. } => {
                (return_impl_trait_id, Some(Arc::clone(&self.allow_async_iterator)))
            }
        };

        let opaque_ty_span =
            self.mark_span_with_reason(DesugaringKind::Async, span, allowed_features);

        let in_trait_or_impl = match fn_kind {
            FnDeclKind::Trait => Some(hir::RpitContext::Trait),
            FnDeclKind::Impl => Some(hir::RpitContext::TraitImpl),
            FnDeclKind::Fn | FnDeclKind::Inherent => None,
            FnDeclKind::ExternFn | FnDeclKind::Closure | FnDeclKind::Pointer => unreachable!(),
        };

        let opaque_ty_ref = self.lower_opaque_inner(
            opaque_ty_node_id,
            hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, in_trait_or_impl },
            opaque_ty_span,
            |this| {
                let bound = this.lower_coroutine_fn_output_type_to_bound(
                    output,
                    coro,
                    opaque_ty_span,
                    ImplTraitContext::OpaqueTy {
                        origin: hir::OpaqueTyOrigin::FnReturn {
                            parent: fn_def_id,
                            in_trait_or_impl,
                        },
                    },
                );
                arena_vec![this; bound]
            },
        );

        let opaque_ty = self.ty(opaque_ty_span, opaque_ty_ref);
        hir::FnRetTy::Return(self.arena.alloc(opaque_ty))
    }

    /// Transforms `-> T` into `Future<Output = T>`.
    fn lower_coroutine_fn_output_type_to_bound(
        &mut self,
        output: &FnRetTy,
        coro: CoroutineKind,
        opaque_ty_span: Span,
        itctx: ImplTraitContext,
    ) -> hir::GenericBound<'hir> {
        // Compute the `T` in `Future<Output = T>` from the return type.
        let output_ty = match output {
            FnRetTy::Ty(ty) => {
                // Not `OpaqueTyOrigin::AsyncFn`: that's only used for the
                // `impl Future` opaque type that `async fn` implicitly
                // generates.
                self.lower_ty(ty, itctx)
            }
            FnRetTy::Default(ret_ty_span) => self.arena.alloc(self.ty_tup(*ret_ty_span, &[])),
        };

        // "<$assoc_ty_name = T>"
        let (assoc_ty_name, trait_lang_item) = match coro {
            CoroutineKind::Async { .. } => (sym::Output, hir::LangItem::Future),
            CoroutineKind::Gen { .. } => (sym::Item, hir::LangItem::Iterator),
            CoroutineKind::AsyncGen { .. } => (sym::Item, hir::LangItem::AsyncIterator),
        };

        let bound_args = self.arena.alloc(hir::GenericArgs {
            args: &[],
            constraints: arena_vec![self; self.assoc_ty_binding(assoc_ty_name, opaque_ty_span, output_ty)],
            parenthesized: hir::GenericArgsParentheses::No,
            span_ext: DUMMY_SP,
        });

        hir::GenericBound::Trait(hir::PolyTraitRef {
            bound_generic_params: &[],
            modifiers: hir::TraitBoundModifiers::NONE,
            trait_ref: hir::TraitRef {
                path: self.make_lang_item_path(trait_lang_item, opaque_ty_span, Some(bound_args)),
                hir_ref_id: self.next_id(),
            },
            span: opaque_ty_span,
        })
    }

    #[instrument(level = "trace", skip(self))]
    fn lower_param_bound(
        &mut self,
        tpb: &GenericBound,
        itctx: ImplTraitContext,
    ) -> hir::GenericBound<'hir> {
        match tpb {
            GenericBound::Trait(p) => hir::GenericBound::Trait(self.lower_poly_trait_ref(p, itctx)),
            GenericBound::Outlives(lifetime) => hir::GenericBound::Outlives(self.lower_lifetime(
                lifetime,
                LifetimeSource::OutlivesBound,
                lifetime.ident.into(),
            )),
            GenericBound::Use(args, span) => hir::GenericBound::Use(
                self.lower_precise_capturing_args(args),
                self.lower_span(*span),
            ),
        }
    }

    fn lower_lifetime(
        &mut self,
        l: &Lifetime,
        source: LifetimeSource,
        syntax: LifetimeSyntax,
    ) -> &'hir hir::Lifetime {
        self.new_named_lifetime(l.id, l.id, l.ident, source, syntax)
    }

    fn lower_lifetime_hidden_in_path(
        &mut self,
        id: NodeId,
        span: Span,
        with_angle_brackets: bool,
    ) -> &'hir hir::Lifetime {
        self.new_named_lifetime(
            id,
            id,
            Ident::new(kw::UnderscoreLifetime, span),
            LifetimeSource::Path { with_angle_brackets },
            LifetimeSyntax::Hidden,
        )
    }

    #[instrument(level = "debug", skip(self))]
    fn new_named_lifetime(
        &mut self,
        id: NodeId,
        new_id: NodeId,
        ident: Ident,
        source: LifetimeSource,
        syntax: LifetimeSyntax,
    ) -> &'hir hir::Lifetime {
        let res = self.resolver.get_lifetime_res(id).unwrap_or(LifetimeRes::Error);
        let res = match res {
            LifetimeRes::Param { param, .. } => hir::LifetimeKind::Param(param),
            LifetimeRes::Fresh { param, .. } => {
                debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
                let param = self.local_def_id(param);
                hir::LifetimeKind::Param(param)
            }
            LifetimeRes::Infer => {
                debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
                hir::LifetimeKind::Infer
            }
            LifetimeRes::Static { .. } => {
                debug_assert!(matches!(ident.name, kw::StaticLifetime | kw::UnderscoreLifetime));
                hir::LifetimeKind::Static
            }
            LifetimeRes::Error => hir::LifetimeKind::Error,
            LifetimeRes::ElidedAnchor { .. } => {
                panic!("Unexpected `ElidedAnchar` {:?} at {:?}", ident, ident.span);
            }
        };

        debug!(?res);
        self.arena.alloc(hir::Lifetime::new(
            self.lower_node_id(new_id),
            self.lower_ident(ident),
            res,
            source,
            syntax,
        ))
    }

    fn lower_generic_params_mut(
        &mut self,
        params: &[GenericParam],
        source: hir::GenericParamSource,
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> {
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
        let (name, kind) = self.lower_generic_param_kind(param, source);

        let hir_id = self.lower_node_id(param.id);
        self.lower_attrs(hir_id, &param.attrs, param.span());
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
        source: hir::GenericParamSource,
    ) -> (hir::ParamName, hir::GenericParamKind<'hir>) {
        match &param.kind {
            GenericParamKind::Lifetime => {
                // AST resolution emitted an error on those parameters, so we lower them using
                // `ParamName::Error`.
                let ident = self.lower_ident(param.ident);
                let param_name =
                    if let Some(LifetimeRes::Error) = self.resolver.get_lifetime_res(param.id) {
                        ParamName::Error(ident)
                    } else {
                        ParamName::Plain(ident)
                    };
                let kind =
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit };

                (param_name, kind)
            }
            GenericParamKind::Type { default, .. } => {
                // Not only do we deny type param defaults in binders but we also map them to `None`
                // since later compiler stages cannot handle them (and shouldn't need to be able to).
                let default = default
                    .as_ref()
                    .filter(|_| match source {
                        hir::GenericParamSource::Generics => true,
                        hir::GenericParamSource::Binder => {
                            self.dcx().emit_err(errors::GenericParamDefaultInBinder {
                                span: param.span(),
                            });

                            false
                        }
                    })
                    .map(|def| {
                        self.lower_ty(
                            def,
                            ImplTraitContext::Disallowed(ImplTraitPosition::GenericDefault),
                        )
                    });

                let kind = hir::GenericParamKind::Type { default, synthetic: false };

                (hir::ParamName::Plain(self.lower_ident(param.ident)), kind)
            }
            GenericParamKind::Const { ty, kw_span: _, default } => {
                let ty = self
                    .lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::GenericDefault));

                // Not only do we deny const param defaults in binders but we also map them to `None`
                // since later compiler stages cannot handle them (and shouldn't need to be able to).
                let default = default
                    .as_ref()
                    .filter(|_| match source {
                        hir::GenericParamSource::Generics => true,
                        hir::GenericParamSource::Binder => {
                            self.dcx().emit_err(errors::GenericParamDefaultInBinder {
                                span: param.span(),
                            });

                            false
                        }
                    })
                    .map(|def| self.lower_anon_const_to_const_arg(def));

                (
                    hir::ParamName::Plain(self.lower_ident(param.ident)),
                    hir::GenericParamKind::Const { ty, default, synthetic: false },
                )
            }
        }
    }

    fn lower_trait_ref(
        &mut self,
        modifiers: ast::TraitBoundModifiers,
        p: &TraitRef,
        itctx: ImplTraitContext,
    ) -> hir::TraitRef<'hir> {
        let path = match self.lower_qpath(
            p.ref_id,
            &None,
            &p.path,
            ParamMode::Explicit,
            AllowReturnTypeNotation::No,
            itctx,
            Some(modifiers),
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
        itctx: ImplTraitContext,
    ) -> hir::PolyTraitRef<'hir> {
        let bound_generic_params =
            self.lower_lifetime_binder(p.trait_ref.ref_id, &p.bound_generic_params);
        let trait_ref = self.lower_trait_ref(p.modifiers, &p.trait_ref, itctx);
        let modifiers = self.lower_trait_bound_modifiers(p.modifiers);
        hir::PolyTraitRef {
            bound_generic_params,
            modifiers,
            trait_ref,
            span: self.lower_span(p.span),
        }
    }

    fn lower_mt(&mut self, mt: &MutTy, itctx: ImplTraitContext) -> hir::MutTy<'hir> {
        hir::MutTy { ty: self.lower_ty(&mt.ty, itctx), mutbl: mt.mutbl }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn lower_param_bounds(
        &mut self,
        bounds: &[GenericBound],
        itctx: ImplTraitContext,
    ) -> hir::GenericBounds<'hir> {
        self.arena.alloc_from_iter(self.lower_param_bounds_mut(bounds, itctx))
    }

    fn lower_param_bounds_mut(
        &mut self,
        bounds: &[GenericBound],
        itctx: ImplTraitContext,
    ) -> impl Iterator<Item = hir::GenericBound<'hir>> {
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
            ImplTraitContext::Universal,
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

    fn lower_array_length_to_const_arg(&mut self, c: &AnonConst) -> &'hir hir::ConstArg<'hir> {
        // We cannot just match on `ExprKind::Underscore` as `(_)` is represented as
        // `ExprKind::Paren(ExprKind::Underscore)` and should also be lowered to `GenericArg::Infer`
        match c.value.peel_parens().kind {
            ExprKind::Underscore => {
                if !self.tcx.features().generic_arg_infer() {
                    feature_err(
                        &self.tcx.sess,
                        sym::generic_arg_infer,
                        c.value.span,
                        fluent_generated::ast_lowering_underscore_array_length_unstable,
                    )
                    .stash(c.value.span, StashKey::UnderscoreForArrayLengths);
                }
                let ct_kind = hir::ConstArgKind::Infer(self.lower_span(c.value.span), ());
                self.arena.alloc(hir::ConstArg { hir_id: self.lower_node_id(c.id), kind: ct_kind })
            }
            _ => self.lower_anon_const_to_const_arg(c),
        }
    }

    /// Used when lowering a type argument that turned out to actually be a const argument.
    ///
    /// Only use for that purpose since otherwise it will create a duplicate def.
    #[instrument(level = "debug", skip(self))]
    fn lower_const_path_to_const_arg(
        &mut self,
        path: &Path,
        res: Res<NodeId>,
        ty_id: NodeId,
        span: Span,
    ) -> &'hir hir::ConstArg<'hir> {
        let tcx = self.tcx;

        let ct_kind = if path
            .is_potential_trivial_const_arg(tcx.features().min_generic_const_args())
            && (tcx.features().min_generic_const_args()
                || matches!(res, Res::Def(DefKind::ConstParam, _)))
        {
            let qpath = self.lower_qpath(
                ty_id,
                &None,
                path,
                ParamMode::Optional,
                AllowReturnTypeNotation::No,
                // FIXME(mgca): update for `fn foo() -> Bar<FOO<impl Trait>>` support
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                None,
            );
            hir::ConstArgKind::Path(qpath)
        } else {
            // Construct an AnonConst where the expr is the "ty"'s path.

            let parent_def_id = self.current_hir_id_owner.def_id;
            let node_id = self.next_node_id();
            let span = self.lower_span(span);

            // Add a definition for the in-band const def.
            // We're lowering a const argument that was originally thought to be a type argument,
            // so the def collector didn't create the def ahead of time. That's why we have to do
            // it here.
            let def_id = self.create_def(parent_def_id, node_id, None, DefKind::AnonConst, span);
            let hir_id = self.lower_node_id(node_id);

            let path_expr = Expr {
                id: ty_id,
                kind: ExprKind::Path(None, path.clone()),
                span,
                attrs: AttrVec::new(),
                tokens: None,
            };

            let ct = self.with_new_scopes(span, |this| {
                self.arena.alloc(hir::AnonConst {
                    def_id,
                    hir_id,
                    body: this.lower_const_body(path_expr.span, Some(&path_expr)),
                    span,
                })
            });
            hir::ConstArgKind::Anon(ct)
        };

        self.arena.alloc(hir::ConstArg { hir_id: self.next_id(), kind: ct_kind })
    }

    /// See [`hir::ConstArg`] for when to use this function vs
    /// [`Self::lower_anon_const_to_anon_const`].
    fn lower_anon_const_to_const_arg(&mut self, anon: &AnonConst) -> &'hir hir::ConstArg<'hir> {
        self.arena.alloc(self.lower_anon_const_to_const_arg_direct(anon))
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_anon_const_to_const_arg_direct(&mut self, anon: &AnonConst) -> hir::ConstArg<'hir> {
        let tcx = self.tcx;
        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = if let ExprKind::Block(block, _) = &anon.value.kind
            && let [stmt] = block.stmts.as_slice()
            && let StmtKind::Expr(expr) = &stmt.kind
            && let ExprKind::Path(..) = &expr.kind
        {
            expr
        } else {
            &anon.value
        };
        let maybe_res =
            self.resolver.get_partial_res(expr.id).and_then(|partial_res| partial_res.full_res());
        if let ExprKind::Path(qself, path) = &expr.kind
            && path.is_potential_trivial_const_arg(tcx.features().min_generic_const_args())
            && (tcx.features().min_generic_const_args()
                || matches!(maybe_res, Some(Res::Def(DefKind::ConstParam, _))))
        {
            let qpath = self.lower_qpath(
                expr.id,
                qself,
                path,
                ParamMode::Optional,
                AllowReturnTypeNotation::No,
                // FIXME(mgca): update for `fn foo() -> Bar<FOO<impl Trait>>` support
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                None,
            );

            return ConstArg {
                hir_id: self.lower_node_id(anon.id),
                kind: hir::ConstArgKind::Path(qpath),
            };
        }

        let lowered_anon = self.lower_anon_const_to_anon_const(anon);
        ConstArg { hir_id: self.next_id(), kind: hir::ConstArgKind::Anon(lowered_anon) }
    }

    /// See [`hir::ConstArg`] for when to use this function vs
    /// [`Self::lower_anon_const_to_const_arg`].
    fn lower_anon_const_to_anon_const(&mut self, c: &AnonConst) -> &'hir hir::AnonConst {
        self.arena.alloc(self.with_new_scopes(c.value.span, |this| {
            let def_id = this.local_def_id(c.id);
            let hir_id = this.lower_node_id(c.id);
            hir::AnonConst {
                def_id,
                hir_id,
                body: this.lower_const_body(c.value.span, Some(&c.value)),
                span: this.lower_span(c.value.span),
            }
        }))
    }

    fn lower_unsafe_source(&mut self, u: UnsafeSource) -> hir::UnsafeSource {
        match u {
            CompilerGenerated => hir::UnsafeSource::CompilerGenerated,
            UserProvided => hir::UnsafeSource::UserProvided,
        }
    }

    fn lower_trait_bound_modifiers(
        &mut self,
        modifiers: TraitBoundModifiers,
    ) -> hir::TraitBoundModifiers {
        hir::TraitBoundModifiers { constness: modifiers.constness, polarity: modifiers.polarity }
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
        attrs: Option<&'hir [hir::Attribute]>,
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
        let local = hir::LetStmt {
            super_: None,
            hir_id,
            init,
            pat,
            els: None,
            source,
            span: self.lower_span(span),
            ty: None,
        };
        self.stmt(span, hir::StmtKind::Let(self.arena.alloc(local)))
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
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowContinue, field)
    }

    fn pat_cf_break(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowBreak, field)
    }

    fn pat_some(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::OptionSome, field)
    }

    fn pat_none(&mut self, span: Span) -> &'hir hir::Pat<'hir> {
        self.pat_lang_item_variant(span, hir::LangItem::OptionNone, &[])
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
    ) -> &'hir hir::Pat<'hir> {
        let qpath = hir::QPath::LangItem(lang_item, self.lower_span(span));
        self.pat(span, hir::PatKind::Struct(qpath, fields, false))
    }

    fn pat_ident(&mut self, span: Span, ident: Ident) -> (&'hir hir::Pat<'hir>, HirId) {
        self.pat_ident_binding_mode(span, ident, hir::BindingMode::NONE)
    }

    fn pat_ident_mut(&mut self, span: Span, ident: Ident) -> (hir::Pat<'hir>, HirId) {
        self.pat_ident_binding_mode_mut(span, ident, hir::BindingMode::NONE)
    }

    fn pat_ident_binding_mode(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingMode,
    ) -> (&'hir hir::Pat<'hir>, HirId) {
        let (pat, hir_id) = self.pat_ident_binding_mode_mut(span, ident, bm);
        (self.arena.alloc(pat), hir_id)
    }

    fn pat_ident_binding_mode_mut(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingMode,
    ) -> (hir::Pat<'hir>, HirId) {
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

    fn ty_path(&mut self, mut hir_id: HirId, span: Span, qpath: hir::QPath<'hir>) -> hir::Ty<'hir> {
        let kind = match qpath {
            hir::QPath::Resolved(None, path) => {
                // Turn trait object paths into `TyKind::TraitObject` instead.
                match path.res {
                    Res::Def(DefKind::Trait | DefKind::TraitAlias, _) => {
                        let principal = hir::PolyTraitRef {
                            bound_generic_params: &[],
                            modifiers: hir::TraitBoundModifiers::NONE,
                            trait_ref: hir::TraitRef { path, hir_ref_id: hir_id },
                            span: self.lower_span(span),
                        };

                        // The original ID is taken by the `PolyTraitRef`,
                        // so the `Ty` itself needs a different one.
                        hir_id = self.next_id();
                        hir::TyKind::TraitObject(
                            arena_vec![self; principal],
                            TaggedRef::new(self.elided_dyn_bound(span), TraitObjectSyntax::None),
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
        let r = hir::Lifetime::new(
            self.next_id(),
            Ident::new(kw::UnderscoreLifetime, self.lower_span(span)),
            hir::LifetimeKind::ImplicitObjectLifetimeDefault,
            LifetimeSource::Other,
            LifetimeSyntax::Hidden,
        );
        debug!("elided_dyn_bound: r={:?}", r);
        self.arena.alloc(r)
    }
}

/// Helper struct for the delayed construction of [`hir::GenericArgs`].
struct GenericArgsCtor<'hir> {
    args: SmallVec<[hir::GenericArg<'hir>; 4]>,
    constraints: &'hir [hir::AssocItemConstraint<'hir>],
    parenthesized: hir::GenericArgsParentheses,
    span: Span,
}

impl<'hir> GenericArgsCtor<'hir> {
    fn is_empty(&self) -> bool {
        self.args.is_empty()
            && self.constraints.is_empty()
            && self.parenthesized == hir::GenericArgsParentheses::No
    }

    fn into_generic_args(self, this: &LoweringContext<'_, 'hir>) -> &'hir hir::GenericArgs<'hir> {
        let ga = hir::GenericArgs {
            args: this.arena.alloc_from_iter(self.args),
            constraints: self.constraints,
            parenthesized: self.parenthesized,
            span_ext: this.lower_span(self.span),
        };
        this.arena.alloc(ga)
    }
}
