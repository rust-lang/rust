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

#![feature(crate_visibility_modifier)]
#![feature(box_patterns)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(never_type)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]

use rustc_ast::tokenstream::{CanSynthesizeMissingTokens, TokenStream};
use rustc_ast::visit;
use rustc_ast::{self as ast, *};
use rustc_ast_pretty::pprust;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace, PartialRes, PerNS, Res};
use rustc_hir::def_id::{DefId, DefPathHash, LocalDefId, CRATE_DEF_ID};
use rustc_hir::definitions::{DefKey, DefPathData, Definitions};
use rustc_hir::{ConstArg, GenericArg, ItemLocalId, ParamName, TraitCandidate};
use rustc_index::vec::{Idx, IndexVec};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::parse::feature_err;
use rustc_session::utils::{FlattenNonterminals, NtToTokenstream};
use rustc_session::Session;
use rustc_span::hygiene::{ExpnId, MacroKind};
use rustc_span::source_map::DesugaringKind;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use tracing::{debug, trace};

macro_rules! arena_vec {
    ($this:expr; $($x:expr),*) => (
        $this.arena.alloc_from_iter([$($x),*])
    );
}

mod asm;
mod block;
mod expr;
mod index;
mod item;
mod pat;
mod path;

rustc_hir::arena_types!(rustc_arena::declare_arena);

struct LoweringContext<'a, 'hir: 'a> {
    /// Used to assign IDs to HIR nodes that do not directly correspond to AST nodes.
    sess: &'a Session,

    resolver: &'a mut dyn ResolverAstLowering,

    /// HACK(Centril): there is a cyclic dependency between the parser and lowering
    /// if we don't have this function pointer. To avoid that dependency so that
    /// `rustc_middle` is independent of the parser, we use dynamic dispatch here.
    nt_to_tokenstream: NtToTokenstream,

    /// Used to allocate HIR nodes.
    arena: &'hir Arena<'hir>,

    /// Bodies inside the owner being lowered.
    bodies: Vec<(hir::ItemLocalId, &'hir hir::Body<'hir>)>,
    /// Attributes inside the owner being lowered.
    attrs: SortedMap<hir::ItemLocalId, &'hir [Attribute]>,
    /// Collect items that were created by lowering the current owner.
    children: FxHashMap<LocalDefId, hir::MaybeOwner<&'hir hir::OwnerInfo<'hir>>>,

    generator_kind: Option<hir::GeneratorKind>,

    /// When inside an `async` context, this is the `HirId` of the
    /// `task_context` local bound to the resume argument of the generator.
    task_context: Option<hir::HirId>,

    /// Used to get the current `fn`'s def span to point to when using `await`
    /// outside of an `async fn`.
    current_item: Option<Span>,

    catch_scope: Option<NodeId>,
    loop_scope: Option<NodeId>,
    is_in_loop_condition: bool,
    is_in_trait_impl: bool,
    is_in_dyn_type: bool,

    /// Used to handle lifetimes appearing in impl-traits.
    captured_lifetimes: Option<LifetimeCaptureContext>,

    current_hir_id_owner: LocalDefId,
    item_local_id_counter: hir::ItemLocalId,
    local_id_to_def_id: SortedMap<ItemLocalId, LocalDefId>,
    trait_map: FxHashMap<ItemLocalId, Box<[TraitCandidate]>>,

    /// NodeIds that are lowered inside the current HIR owner.
    node_id_to_local_id: FxHashMap<NodeId, hir::ItemLocalId>,

    allow_try_trait: Option<Lrc<[Symbol]>>,
    allow_gen_future: Option<Lrc<[Symbol]>>,
    allow_into_future: Option<Lrc<[Symbol]>>,
}

/// Resolution for a lifetime appearing in a type.
#[derive(Copy, Clone, Debug)]
pub enum LifetimeRes {
    /// Successfully linked the lifetime to a generic parameter.
    Param {
        /// Id of the generic parameter that introduced it.
        param: LocalDefId,
        /// Id of the introducing place. That can be:
        /// - an item's id, for the item's generic parameters;
        /// - a TraitRef's ref_id, identifying the `for<...>` binder;
        /// - a BareFn type's id;
        /// - a Path's id when this path has parenthesized generic args.
        ///
        /// This information is used for impl-trait lifetime captures, to know when to or not to
        /// capture any given lifetime.
        binder: NodeId,
    },
    /// Created a generic parameter for an anonymous lifetime.
    Fresh {
        /// Id of the generic parameter that introduced it.
        param: LocalDefId,
        /// Id of the introducing place. See `Param`.
        binder: NodeId,
    },
    /// This variant is used for anonymous lifetimes that we did not resolve during
    /// late resolution.  Shifting the work to the HIR lifetime resolver.
    Anonymous {
        /// Id of the introducing place. See `Param`.
        binder: NodeId,
        /// Whether this lifetime was spelled or elided.
        elided: bool,
    },
    /// Explicit `'static` lifetime.
    Static,
    /// Resolution failure.
    Error,
    /// HACK: This is used to recover the NodeId of an elided lifetime.
    ElidedAnchor { start: NodeId, end: NodeId },
}

/// When we lower a lifetime, it is inserted in `captures`, and the resolution is modified so
/// to point to the lifetime parameter impl-trait will generate.
/// When traversing `for<...>` binders, they are inserted in `binders_to_ignore` so we know *not*
/// to rebind the introduced lifetimes.
#[derive(Debug)]
struct LifetimeCaptureContext {
    /// parent def_id for new definitions
    parent_def_id: LocalDefId,
    /// Set of lifetimes to rebind.
    captures: FxHashMap<
        LocalDefId, // original parameter id
        (
            Span,        // Span
            NodeId,      // synthetized parameter id
            ParamName,   // parameter name
            LifetimeRes, // original resolution
        ),
    >,
    /// Traversed binders.  The ids in this set should *not* be rebound.
    binders_to_ignore: FxHashSet<NodeId>,
}

pub trait ResolverAstLowering {
    fn def_key(&self, id: DefId) -> DefKey;

    fn def_span(&self, id: LocalDefId) -> Span;

    fn item_generics_num_lifetimes(&self, def: DefId) -> usize;

    fn legacy_const_generic_args(&mut self, expr: &Expr) -> Option<Vec<usize>>;

    /// Obtains resolution for a `NodeId` with a single resolution.
    fn get_partial_res(&self, id: NodeId) -> Option<PartialRes>;

    /// Obtains per-namespace resolutions for `use` statement with the given `NodeId`.
    fn get_import_res(&self, id: NodeId) -> PerNS<Option<Res<NodeId>>>;

    /// Obtains resolution for a label with the given `NodeId`.
    fn get_label_res(&self, id: NodeId) -> Option<NodeId>;

    /// Obtains resolution for a lifetime with the given `NodeId`.
    fn get_lifetime_res(&self, id: NodeId) -> Option<LifetimeRes>;

    /// Obtain the list of lifetimes parameters to add to an item.
    fn take_extra_lifetime_params(&mut self, id: NodeId) -> Vec<(Ident, NodeId, LifetimeRes)>;

    fn create_stable_hashing_context(&self) -> StableHashingContext<'_>;

    fn definitions(&self) -> &Definitions;

    fn next_node_id(&mut self) -> NodeId;

    fn take_trait_map(&mut self, node: NodeId) -> Option<Vec<hir::TraitCandidate>>;

    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId>;

    fn local_def_id(&self, node: NodeId) -> LocalDefId;

    fn def_path_hash(&self, def_id: DefId) -> DefPathHash;

    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        data: DefPathData,
        expn_id: ExpnId,
        span: Span,
    ) -> LocalDefId;

    fn decl_macro_kind(&self, def_id: LocalDefId) -> MacroKind;
}

/// Context of `impl Trait` in code, which determines whether it is allowed in an HIR subtree,
/// and if so, what meaning it has.
#[derive(Debug)]
enum ImplTraitContext<'b, 'a> {
    /// Treat `impl Trait` as shorthand for a new universal generic parameter.
    /// Example: `fn foo(x: impl Debug)`, where `impl Debug` is conceptually
    /// equivalent to a fresh universal parameter like `fn foo<T: Debug>(x: T)`.
    ///
    /// Newly generated parameters should be inserted into the given `Vec`.
    Universal(&'b mut Vec<hir::GenericParam<'a>>, &'b mut Vec<hir::WherePredicate<'a>>, LocalDefId),

    /// Treat `impl Trait` as shorthand for a new opaque type.
    /// Example: `fn foo() -> impl Debug`, where `impl Debug` is conceptually
    /// equivalent to a new opaque type like `type T = impl Debug; fn foo() -> T`.
    ///
    ReturnPositionOpaqueTy {
        /// Origin: Either OpaqueTyOrigin::FnReturn or OpaqueTyOrigin::AsyncFn,
        origin: hir::OpaqueTyOrigin,
    },
    /// Impl trait in type aliases.
    TypeAliasesOpaqueTy,
    /// `impl Trait` is not accepted in this position.
    Disallowed(ImplTraitPosition),
}

/// Position in which `impl Trait` is disallowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitPosition {
    Path,
    Variable,
    Type,
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
    TraitReturn,
    ImplReturn,
}

impl<'a> ImplTraitContext<'_, 'a> {
    fn reborrow<'this>(&'this mut self) -> ImplTraitContext<'this, 'a> {
        use self::ImplTraitContext::*;
        match self {
            Universal(params, bounds, parent) => Universal(params, bounds, *parent),
            ReturnPositionOpaqueTy { origin } => ReturnPositionOpaqueTy { origin: *origin },
            TypeAliasesOpaqueTy => TypeAliasesOpaqueTy,
            Disallowed(pos) => Disallowed(*pos),
        }
    }
}

impl std::fmt::Display for ImplTraitPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ImplTraitPosition::Path => "path",
            ImplTraitPosition::Variable => "variable binding",
            ImplTraitPosition::Type => "type",
            ImplTraitPosition::Trait => "trait",
            ImplTraitPosition::AsyncBlock => "async block",
            ImplTraitPosition::Bound => "bound",
            ImplTraitPosition::Generic => "generic",
            ImplTraitPosition::ExternFnParam => "`extern fn` param",
            ImplTraitPosition::ClosureParam => "closure param",
            ImplTraitPosition::PointerParam => "`fn` pointer param",
            ImplTraitPosition::FnTraitParam => "`Fn` trait param",
            ImplTraitPosition::TraitParam => "trait method param",
            ImplTraitPosition::ImplParam => "`impl` method param",
            ImplTraitPosition::ExternFnReturn => "`extern fn` return",
            ImplTraitPosition::ClosureReturn => "closure return",
            ImplTraitPosition::PointerReturn => "`fn` pointer return",
            ImplTraitPosition::FnTraitReturn => "`Fn` trait return",
            ImplTraitPosition::TraitReturn => "trait method return",
            ImplTraitPosition::ImplReturn => "`impl` method return",
        };

        write!(f, "{}", name)
    }
}

#[derive(Debug)]
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
    fn impl_trait_return_allowed(&self) -> bool {
        match self {
            FnDeclKind::Fn | FnDeclKind::Inherent => true,
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
    resolver: &dyn ResolverAstLowering,
    krate: &'a Crate,
) -> IndexVec<LocalDefId, AstOwner<'a>> {
    let mut indexer = Indexer { resolver, index: IndexVec::new() };
    indexer.index.ensure_contains_elem(CRATE_DEF_ID, || AstOwner::NonOwner);
    indexer.index[CRATE_DEF_ID] = AstOwner::Crate(krate);
    visit::walk_crate(&mut indexer, krate);
    return indexer.index;

    struct Indexer<'s, 'a> {
        resolver: &'s dyn ResolverAstLowering,
        index: IndexVec<LocalDefId, AstOwner<'a>>,
    }

    impl<'a> visit::Visitor<'a> for Indexer<'_, 'a> {
        fn visit_attribute(&mut self, _: &'a Attribute) {
            // We do not want to lower expressions that appear in attributes,
            // as they are not accessible to the rest of the HIR.
        }

        fn visit_item(&mut self, item: &'a ast::Item) {
            let def_id = self.resolver.local_def_id(item.id);
            self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner);
            self.index[def_id] = AstOwner::Item(item);
            visit::walk_item(self, item)
        }

        fn visit_assoc_item(&mut self, item: &'a ast::AssocItem, ctxt: visit::AssocCtxt) {
            let def_id = self.resolver.local_def_id(item.id);
            self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner);
            self.index[def_id] = AstOwner::AssocItem(item, ctxt);
            visit::walk_assoc_item(self, item, ctxt);
        }

        fn visit_foreign_item(&mut self, item: &'a ast::ForeignItem) {
            let def_id = self.resolver.local_def_id(item.id);
            self.index.ensure_contains_elem(def_id, || AstOwner::NonOwner);
            self.index[def_id] = AstOwner::ForeignItem(item);
            visit::walk_foreign_item(self, item);
        }
    }
}

/// Compute the hash for the HIR of the full crate.
/// This hash will then be part of the crate_hash which is stored in the metadata.
fn compute_hir_hash(
    resolver: &mut dyn ResolverAstLowering,
    owners: &IndexVec<LocalDefId, hir::MaybeOwner<&hir::OwnerInfo<'_>>>,
) -> Fingerprint {
    let mut hir_body_nodes: Vec<_> = owners
        .iter_enumerated()
        .filter_map(|(def_id, info)| {
            let info = info.as_owner()?;
            let def_path_hash = resolver.definitions().def_path_hash(def_id);
            Some((def_path_hash, info))
        })
        .collect();
    hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

    let mut stable_hasher = StableHasher::new();
    let mut hcx = resolver.create_stable_hashing_context();
    hir_body_nodes.hash_stable(&mut hcx, &mut stable_hasher);
    stable_hasher.finish()
}

pub fn lower_crate<'a, 'hir>(
    sess: &'a Session,
    krate: &'a Crate,
    resolver: &'a mut dyn ResolverAstLowering,
    nt_to_tokenstream: NtToTokenstream,
    arena: &'hir Arena<'hir>,
) -> &'hir hir::Crate<'hir> {
    let _prof_timer = sess.prof.verbose_generic_activity("hir_lowering");

    let ast_index = index_crate(resolver, krate);

    let mut owners =
        IndexVec::from_fn_n(|_| hir::MaybeOwner::Phantom, resolver.definitions().def_index_count());

    for def_id in ast_index.indices() {
        item::ItemLowerer {
            sess,
            resolver,
            nt_to_tokenstream,
            arena,
            ast_index: &ast_index,
            owners: &mut owners,
        }
        .lower_node(def_id);
    }

    let hir_hash = compute_hir_hash(resolver, &owners);
    let krate = hir::Crate { owners, hir_hash };
    arena.alloc(krate)
}

#[derive(Copy, Clone, PartialEq)]
enum ParamMode {
    /// Any path in a type context.
    Explicit,
    /// Path in a type definition, where the anonymous lifetime `'_` is not allowed.
    ExplicitNamed,
    /// The `module::Type` in `module::Type::method` in an expression.
    Optional,
}

enum ParenthesizedGenericArgs {
    Ok,
    Err,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    fn with_hir_id_owner(
        &mut self,
        owner: NodeId,
        f: impl FnOnce(&mut Self) -> hir::OwnerNode<'hir>,
    ) {
        let def_id = self.resolver.local_def_id(owner);

        let current_attrs = std::mem::take(&mut self.attrs);
        let current_bodies = std::mem::take(&mut self.bodies);
        let current_node_ids = std::mem::take(&mut self.node_id_to_local_id);
        let current_id_to_def_id = std::mem::take(&mut self.local_id_to_def_id);
        let current_trait_map = std::mem::take(&mut self.trait_map);
        let current_owner = std::mem::replace(&mut self.current_hir_id_owner, def_id);
        let current_local_counter =
            std::mem::replace(&mut self.item_local_id_counter, hir::ItemLocalId::new(1));

        // Always allocate the first `HirId` for the owner itself.
        let _old = self.node_id_to_local_id.insert(owner, hir::ItemLocalId::new(0));
        debug_assert_eq!(_old, None);

        let item = f(self);
        debug_assert_eq!(def_id, item.def_id());
        let info = self.make_owner_info(item);

        self.attrs = current_attrs;
        self.bodies = current_bodies;
        self.node_id_to_local_id = current_node_ids;
        self.local_id_to_def_id = current_id_to_def_id;
        self.trait_map = current_trait_map;
        self.current_hir_id_owner = current_owner;
        self.item_local_id_counter = current_local_counter;

        let _old = self.children.insert(def_id, hir::MaybeOwner::Owner(info));
        debug_assert!(_old.is_none())
    }

    fn make_owner_info(&mut self, node: hir::OwnerNode<'hir>) -> &'hir hir::OwnerInfo<'hir> {
        let attrs = std::mem::take(&mut self.attrs);
        let mut bodies = std::mem::take(&mut self.bodies);
        let local_id_to_def_id = std::mem::take(&mut self.local_id_to_def_id);
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
        let (hash_including_bodies, hash_without_bodies) = self.hash_owner(node, &bodies);
        let (nodes, parenting) =
            index::index_hir(self.sess, self.resolver.definitions(), node, &bodies);
        let nodes = hir::OwnerNodes {
            hash_including_bodies,
            hash_without_bodies,
            nodes,
            bodies,
            local_id_to_def_id,
        };
        let attrs = {
            let mut hcx = self.resolver.create_stable_hashing_context();
            let mut stable_hasher = StableHasher::new();
            attrs.hash_stable(&mut hcx, &mut stable_hasher);
            let hash = stable_hasher.finish();
            hir::AttributeMap { map: attrs, hash }
        };

        self.arena.alloc(hir::OwnerInfo { nodes, parenting, attrs, trait_map })
    }

    /// Hash the HIR node twice, one deep and one shallow hash.  This allows to differentiate
    /// queries which depend on the full HIR tree and those which only depend on the item signature.
    fn hash_owner(
        &mut self,
        node: hir::OwnerNode<'hir>,
        bodies: &SortedMap<hir::ItemLocalId, &'hir hir::Body<'hir>>,
    ) -> (Fingerprint, Fingerprint) {
        let mut hcx = self.resolver.create_stable_hashing_context();
        let mut stable_hasher = StableHasher::new();
        hcx.with_hir_bodies(true, node.def_id(), bodies, |hcx| {
            node.hash_stable(hcx, &mut stable_hasher)
        });
        let hash_including_bodies = stable_hasher.finish();
        let mut stable_hasher = StableHasher::new();
        hcx.with_hir_bodies(false, node.def_id(), bodies, |hcx| {
            node.hash_stable(hcx, &mut stable_hasher)
        });
        let hash_without_bodies = stable_hasher.finish();
        (hash_including_bodies, hash_without_bodies)
    }

    /// This method allocates a new `HirId` for the given `NodeId` and stores it in
    /// the `LoweringContext`'s `NodeId => HirId` map.
    /// Take care not to call this method if the resulting `HirId` is then not
    /// actually used in the HIR, as that would trigger an assertion in the
    /// `HirIdValidator` later on, which makes sure that all `NodeId`s got mapped
    /// properly. Calling the method twice with the same `NodeId` is fine though.
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
                if let Some(def_id) = self.resolver.opt_local_def_id(ast_node_id) {
                    // Do not override a `MaybeOwner::Owner` that may already here.
                    self.children.entry(def_id).or_insert(hir::MaybeOwner::NonOwner(hir_id));
                    self.local_id_to_def_id.insert(local_id, def_id);
                }

                if let Some(traits) = self.resolver.take_trait_map(ast_node_id) {
                    self.trait_map.insert(hir_id.local_id, traits.into_boxed_slice());
                }

                hir_id
            }
        }
    }

    fn next_id(&mut self) -> hir::HirId {
        let node_id = self.resolver.next_node_id();
        self.lower_node_id(node_id)
    }

    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        let res: Result<Res, ()> = res.apply_id(|id| {
            let owner = self.current_hir_id_owner;
            let local_id = self.node_id_to_local_id.get(&id).copied().ok_or(())?;
            Ok(hir::HirId { owner, local_id })
        });
        // We may fail to find a HirId when the Res points to a Local from an enclosing HIR owner.
        // This can happen when trying to lower the return type `x` in erroneous code like
        //   async fn foo(x: u8) -> x {}
        // In that case, `x` is lowered as a function parameter, and the return type is lowered as
        // an opaque type as a synthesized HIR owner.
        res.unwrap_or(Res::Err)
    }

    fn expect_full_res(&mut self, id: NodeId) -> Res<NodeId> {
        self.resolver.get_partial_res(id).map_or(Res::Err, |pr| {
            if pr.unresolved_segments() != 0 {
                panic!("path not fully resolved: {:?}", pr);
            }
            pr.base_res()
        })
    }

    fn expect_full_res_from_use(&mut self, id: NodeId) -> impl Iterator<Item = Res<NodeId>> {
        self.resolver.get_import_res(id).present_items()
    }

    fn diagnostic(&self) -> &rustc_errors::Handler {
        self.sess.diagnostic()
    }

    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    fn mark_span_with_reason(
        &self,
        reason: DesugaringKind,
        span: Span,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
    ) -> Span {
        span.mark_with_reason(
            allow_internal_unstable,
            reason,
            self.sess.edition(),
            self.resolver.create_stable_hashing_context(),
        )
    }

    /// Intercept all spans entering HIR.
    /// Mark a span as relative to the current owning item.
    fn lower_span(&self, span: Span) -> Span {
        if self.sess.opts.debugging_opts.incremental_relative_spans {
            span.with_parent(Some(self.current_hir_id_owner))
        } else {
            // Do not make spans relative when not using incremental compilation.
            span
        }
    }

    fn lower_ident(&self, ident: Ident) -> Ident {
        Ident::new(ident.name, self.lower_span(ident.span))
    }

    /// Converts a lifetime into a new generic parameter.
    fn lifetime_res_to_generic_param(
        &mut self,
        ident: Ident,
        node_id: NodeId,
        res: LifetimeRes,
    ) -> Option<hir::GenericParam<'hir>> {
        let (name, kind) = match res {
            LifetimeRes::Param { .. } => {
                (hir::ParamName::Plain(ident), hir::LifetimeParamKind::Explicit)
            }
            LifetimeRes::Fresh { param, .. } => {
                (hir::ParamName::Fresh(param), hir::LifetimeParamKind::Elided)
            }
            LifetimeRes::Static | LifetimeRes::Error => return None,
            res => panic!(
                "Unexpected lifetime resolution {:?} for {:?} at {:?}",
                res, ident, ident.span
            ),
        };
        let hir_id = self.lower_node_id(node_id);
        Some(hir::GenericParam {
            hir_id,
            name,
            span: self.lower_span(ident.span),
            pure_wrt_drop: false,
            kind: hir::GenericParamKind::Lifetime { kind },
            colon_span: None,
        })
    }

    /// Creates a new `hir::GenericParam` for every new `Fresh` lifetime and
    /// universal `impl Trait` type parameter encountered while evaluating `f`.
    /// Definitions are created with the provided `parent_def_id`.
    fn add_implicit_generics<T>(
        &mut self,
        generics: &Generics,
        parent_node_id: NodeId,
        f: impl FnOnce(
            &mut Self,
            &mut Vec<hir::GenericParam<'hir>>,
            &mut Vec<hir::WherePredicate<'hir>>,
        ) -> T,
    ) -> (&'hir hir::Generics<'hir>, T) {
        let mut impl_trait_defs = Vec::new();
        let mut impl_trait_bounds = Vec::new();
        let mut lowered_generics = self.lower_generics_mut(
            generics,
            ImplTraitContext::Universal(
                &mut impl_trait_defs,
                &mut impl_trait_bounds,
                self.current_hir_id_owner,
            ),
        );
        let res = f(self, &mut impl_trait_defs, &mut impl_trait_bounds);

        let extra_lifetimes = self.resolver.take_extra_lifetime_params(parent_node_id);
        lowered_generics.params.extend(
            extra_lifetimes
                .into_iter()
                .filter_map(|(ident, node_id, res)| {
                    self.lifetime_res_to_generic_param(ident, node_id, res)
                })
                .chain(impl_trait_defs),
        );
        lowered_generics.predicates.extend(impl_trait_bounds);

        let lowered_generics = lowered_generics.into_generics(self.arena);
        (lowered_generics, res)
    }

    /// Setup lifetime capture for and impl-trait.
    /// The captures will be added to `captures`.
    fn while_capturing_lifetimes<T>(
        &mut self,
        parent_def_id: LocalDefId,
        captures: &mut FxHashMap<LocalDefId, (Span, NodeId, ParamName, LifetimeRes)>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let lifetime_stash = std::mem::replace(
            &mut self.captured_lifetimes,
            Some(LifetimeCaptureContext {
                parent_def_id,
                captures: std::mem::take(captures),
                binders_to_ignore: Default::default(),
            }),
        );

        let ret = f(self);

        let ctxt = std::mem::replace(&mut self.captured_lifetimes, lifetime_stash).unwrap();
        *captures = ctxt.captures;

        ret
    }

    /// Register a binder to be ignored for lifetime capture.
    #[tracing::instrument(level = "debug", skip(self, f))]
    #[inline]
    fn with_lifetime_binder<T>(&mut self, binder: NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        if let Some(ctxt) = &mut self.captured_lifetimes {
            ctxt.binders_to_ignore.insert(binder);
        }
        let ret = f(self);
        if let Some(ctxt) = &mut self.captured_lifetimes {
            ctxt.binders_to_ignore.remove(&binder);
        }
        ret
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
            AttrKind::Normal(ref item, _) => AttrKind::Normal(
                AttrItem {
                    path: item.path.clone(),
                    args: self.lower_mac_args(&item.args),
                    tokens: None,
                },
                None,
            ),
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

    fn lower_mac_args(&self, args: &MacArgs) -> MacArgs {
        match *args {
            MacArgs::Empty => MacArgs::Empty,
            MacArgs::Delimited(dspan, delim, ref tokens) => {
                // This is either a non-key-value attribute, or a `macro_rules!` body.
                // We either not have any nonterminals present (in the case of an attribute),
                // or have tokens available for all nonterminals in the case of a nested
                // `macro_rules`: e.g:
                //
                // ```rust
                // macro_rules! outer {
                //     ($e:expr) => {
                //         macro_rules! inner {
                //             () => { $e }
                //         }
                //     }
                // }
                // ```
                //
                // In both cases, we don't want to synthesize any tokens
                MacArgs::Delimited(
                    dspan,
                    delim,
                    self.lower_token_stream(tokens.clone(), CanSynthesizeMissingTokens::No),
                )
            }
            // This is an inert key-value attribute - it will never be visible to macros
            // after it gets lowered to HIR. Therefore, we can extract literals to handle
            // nonterminals in `#[doc]` (e.g. `#[doc = $e]`).
            MacArgs::Eq(eq_span, MacArgsEq::Ast(ref expr)) => {
                // In valid code the value always ends up as a single literal. Otherwise, a dummy
                // literal suffices because the error is handled elsewhere.
                let lit = if let ExprKind::Lit(lit) = &expr.kind {
                    lit.clone()
                } else {
                    Lit {
                        token: token::Lit::new(token::LitKind::Err, kw::Empty, None),
                        kind: LitKind::Err(kw::Empty),
                        span: DUMMY_SP,
                    }
                };
                MacArgs::Eq(eq_span, MacArgsEq::Hir(lit))
            }
            MacArgs::Eq(_, MacArgsEq::Hir(ref lit)) => {
                unreachable!("in literal form when lowering mac args eq: {:?}", lit)
            }
        }
    }

    fn lower_token_stream(
        &self,
        tokens: TokenStream,
        synthesize_tokens: CanSynthesizeMissingTokens,
    ) -> TokenStream {
        FlattenNonterminals {
            parse_sess: &self.sess.parse_sess,
            synthesize_tokens,
            nt_to_tokenstream: self.nt_to_tokenstream,
        }
        .process_token_stream(tokens)
    }

    /// Given an associated type constraint like one of these:
    ///
    /// ```
    /// T: Iterator<Item: Debug>
    ///             ^^^^^^^^^^^
    /// T: Iterator<Item = Debug>
    ///             ^^^^^^^^^^^^
    /// ```
    ///
    /// returns a `hir::TypeBinding` representing `Item`.
    fn lower_assoc_ty_constraint(
        &mut self,
        constraint: &AssocConstraint,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::TypeBinding<'hir> {
        debug!("lower_assoc_ty_constraint(constraint={:?}, itctx={:?})", constraint, itctx);

        // lower generic arguments of identifier in constraint
        let gen_args = if let Some(ref gen_args) = constraint.gen_args {
            let gen_args_ctor = match gen_args {
                GenericArgs::AngleBracketed(ref data) => {
                    self.lower_angle_bracketed_parameter_data(
                        data,
                        ParamMode::Explicit,
                        itctx.reborrow(),
                    )
                    .0
                }
                GenericArgs::Parenthesized(ref data) => {
                    let mut err = self.sess.struct_span_err(
                        gen_args.span(),
                        "parenthesized generic arguments cannot be used in associated type constraints"
                    );
                    // FIXME: try to write a suggestion here
                    err.emit();
                    self.lower_angle_bracketed_parameter_data(
                        &data.as_angle_bracketed_args(),
                        ParamMode::Explicit,
                        itctx.reborrow(),
                    )
                    .0
                }
            };
            gen_args_ctor.into_generic_args(self)
        } else {
            self.arena.alloc(hir::GenericArgs::none())
        };

        let kind = match constraint.kind {
            AssocConstraintKind::Equality { ref term } => {
                let term = match term {
                    Term::Ty(ref ty) => self.lower_ty(ty, itctx).into(),
                    Term::Const(ref c) => self.lower_anon_const(c).into(),
                };
                hir::TypeBindingKind::Equality { term }
            }
            AssocConstraintKind::Bound { ref bounds } => {
                let mut parent_def_id = self.current_hir_id_owner;
                // Piggy-back on the `impl Trait` context to figure out the correct behavior.
                let (desugar_to_impl_trait, itctx) = match itctx {
                    // We are in the return position:
                    //
                    //     fn foo() -> impl Iterator<Item: Debug>
                    //
                    // so desugar to
                    //
                    //     fn foo() -> impl Iterator<Item = impl Debug>
                    ImplTraitContext::ReturnPositionOpaqueTy { .. }
                    | ImplTraitContext::TypeAliasesOpaqueTy { .. } => (true, itctx),

                    // We are in the argument position, but within a dyn type:
                    //
                    //     fn foo(x: dyn Iterator<Item: Debug>)
                    //
                    // so desugar to
                    //
                    //     fn foo(x: dyn Iterator<Item = impl Debug>)
                    ImplTraitContext::Universal(_, _, parent) if self.is_in_dyn_type => {
                        parent_def_id = parent;
                        (true, itctx)
                    }

                    // In `type Foo = dyn Iterator<Item: Debug>` we desugar to
                    // `type Foo = dyn Iterator<Item = impl Debug>` but we have to override the
                    // "impl trait context" to permit `impl Debug` in this position (it desugars
                    // then to an opaque type).
                    //
                    // FIXME: this is only needed until `impl Trait` is allowed in type aliases.
                    ImplTraitContext::Disallowed(_) if self.is_in_dyn_type => {
                        (true, ImplTraitContext::TypeAliasesOpaqueTy)
                    }

                    // We are in the parameter position, but not within a dyn type:
                    //
                    //     fn foo(x: impl Iterator<Item: Debug>)
                    //
                    // so we leave it as is and this gets expanded in astconv to a bound like
                    // `<T as Iterator>::Item: Debug` where `T` is the type parameter for the
                    // `impl Iterator`.
                    _ => (false, itctx),
                };

                if desugar_to_impl_trait {
                    // Desugar `AssocTy: Bounds` into `AssocTy = impl Bounds`. We do this by
                    // constructing the HIR for `impl bounds...` and then lowering that.

                    let impl_trait_node_id = self.resolver.next_node_id();
                    self.resolver.create_def(
                        parent_def_id,
                        impl_trait_node_id,
                        DefPathData::ImplTrait,
                        ExpnId::root(),
                        constraint.span,
                    );

                    self.with_dyn_type_scope(false, |this| {
                        let node_id = this.resolver.next_node_id();
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
                } else {
                    // Desugar `AssocTy: Bounds` into a type binding where the
                    // later desugars into a trait predicate.
                    let bounds = self.lower_param_bounds(bounds, itctx);

                    hir::TypeBindingKind::Constraint { bounds }
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

    fn lower_generic_arg(
        &mut self,
        arg: &ast::GenericArg,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericArg<'hir> {
        match arg {
            ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.lower_lifetime(&lt)),
            ast::GenericArg::Type(ty) => {
                match ty.kind {
                    TyKind::Infer if self.sess.features_untracked().generic_arg_infer => {
                        return GenericArg::Infer(hir::InferArg {
                            hir_id: self.lower_node_id(ty.id),
                            span: self.lower_span(ty.span),
                        });
                    }
                    // We parse const arguments as path types as we cannot distinguish them during
                    // parsing. We try to resolve that ambiguity by attempting resolution in both the
                    // type and value namespaces. If we resolved the path in the value namespace, we
                    // transform it into a generic const argument.
                    TyKind::Path(ref qself, ref path) => {
                        if let Some(partial_res) = self.resolver.get_partial_res(ty.id) {
                            let res = partial_res.base_res();
                            if !res.matches_ns(Namespace::TypeNS) {
                                debug!(
                                    "lower_generic_arg: Lowering type argument as const argument: {:?}",
                                    ty,
                                );

                                // Construct an AnonConst where the expr is the "ty"'s path.

                                let parent_def_id = self.current_hir_id_owner;
                                let node_id = self.resolver.next_node_id();

                                // Add a definition for the in-band const def.
                                self.resolver.create_def(
                                    parent_def_id,
                                    node_id,
                                    DefPathData::AnonConst,
                                    ExpnId::root(),
                                    ty.span,
                                );

                                let span = self.lower_span(ty.span);
                                let path_expr = Expr {
                                    id: ty.id,
                                    kind: ExprKind::Path(qself.clone(), path.clone()),
                                    span,
                                    attrs: AttrVec::new(),
                                    tokens: None,
                                };

                                let ct = self.with_new_scopes(|this| hir::AnonConst {
                                    hir_id: this.lower_node_id(node_id),
                                    body: this.lower_const_body(path_expr.span, Some(&path_expr)),
                                });
                                return GenericArg::Const(ConstArg { value: ct, span });
                            }
                        }
                    }
                    _ => {}
                }
                GenericArg::Type(self.lower_ty_direct(&ty, itctx))
            }
            ast::GenericArg::Const(ct) => GenericArg::Const(ConstArg {
                value: self.lower_anon_const(&ct),
                span: self.lower_span(ct.value.span),
            }),
        }
    }

    fn lower_ty(&mut self, t: &Ty, itctx: ImplTraitContext<'_, 'hir>) -> &'hir hir::Ty<'hir> {
        self.arena.alloc(self.lower_ty_direct(t, itctx))
    }

    fn lower_path_ty(
        &mut self,
        t: &Ty,
        qself: &Option<QSelf>,
        path: &Path,
        param_mode: ParamMode,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::Ty<'hir> {
        let id = self.lower_node_id(t.id);
        let qpath = self.lower_qpath(t.id, qself, path, param_mode, itctx);
        self.ty_path(id, t.span, qpath)
    }

    fn ty(&mut self, span: Span, kind: hir::TyKind<'hir>) -> hir::Ty<'hir> {
        hir::Ty { hir_id: self.next_id(), kind, span: self.lower_span(span) }
    }

    fn ty_tup(&mut self, span: Span, tys: &'hir [hir::Ty<'hir>]) -> hir::Ty<'hir> {
        self.ty(span, hir::TyKind::Tup(tys))
    }

    fn lower_ty_direct(&mut self, t: &Ty, mut itctx: ImplTraitContext<'_, 'hir>) -> hir::Ty<'hir> {
        let kind = match t.kind {
            TyKind::Infer => hir::TyKind::Infer,
            TyKind::Err => hir::TyKind::Err,
            TyKind::Slice(ref ty) => hir::TyKind::Slice(self.lower_ty(ty, itctx)),
            TyKind::Ptr(ref mt) => hir::TyKind::Ptr(self.lower_mt(mt, itctx)),
            TyKind::Rptr(ref region, ref mt) => {
                let region = region.unwrap_or_else(|| {
                    let Some(LifetimeRes::ElidedAnchor { start, end }) = self.resolver.get_lifetime_res(t.id) else {
                        panic!()
                    };
                    debug_assert_eq!(start.plus(1), end);
                    let span = self.sess.source_map().next_point(t.span.shrink_to_lo());
                    Lifetime {
                        ident: Ident::new(kw::UnderscoreLifetime, span),
                        id: start,
                    }
                });
                let lifetime = self.lower_lifetime(&region);
                hir::TyKind::Rptr(lifetime, self.lower_mt(mt, itctx))
            }
            TyKind::BareFn(ref f) => self.with_lifetime_binder(t.id, |this| {
                hir::TyKind::BareFn(this.arena.alloc(hir::BareFnTy {
                    generic_params: this.lower_generic_params(&f.generic_params),
                    unsafety: this.lower_unsafety(f.unsafety),
                    abi: this.lower_extern(f.ext),
                    decl: this.lower_fn_decl(&f.decl, None, FnDeclKind::Pointer, None),
                    param_names: this.lower_fn_params_to_names(&f.decl),
                }))
            }),
            TyKind::Never => hir::TyKind::Never,
            TyKind::Tup(ref tys) => {
                hir::TyKind::Tup(self.arena.alloc_from_iter(
                    tys.iter().map(|ty| self.lower_ty_direct(ty, itctx.reborrow())),
                ))
            }
            TyKind::Paren(ref ty) => {
                return self.lower_ty_direct(ty, itctx);
            }
            TyKind::Path(ref qself, ref path) => {
                return self.lower_path_ty(t, qself, path, ParamMode::Explicit, itctx);
            }
            TyKind::ImplicitSelf => {
                let res = self.expect_full_res(t.id);
                let res = self.lower_res(res);
                hir::TyKind::Path(hir::QPath::Resolved(
                    None,
                    self.arena.alloc(hir::Path {
                        res,
                        segments: arena_vec![self; hir::PathSegment::from_ident(
                            Ident::with_dummy_span(kw::SelfUpper)
                        )],
                        span: self.lower_span(t.span),
                    }),
                ))
            }
            TyKind::Array(ref ty, ref length) => {
                hir::TyKind::Array(self.lower_ty(ty, itctx), self.lower_array_length(length))
            }
            TyKind::Typeof(ref expr) => hir::TyKind::Typeof(self.lower_anon_const(expr)),
            TyKind::TraitObject(ref bounds, kind) => {
                let mut lifetime_bound = None;
                let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                    let bounds =
                        this.arena.alloc_from_iter(bounds.iter().filter_map(
                            |bound| match *bound {
                                GenericBound::Trait(
                                    ref ty,
                                    TraitBoundModifier::None | TraitBoundModifier::MaybeConst,
                                ) => Some(this.lower_poly_trait_ref(ty, itctx.reborrow())),
                                // `~const ?Bound` will cause an error during AST validation
                                // anyways, so treat it like `?Bound` as compilation proceeds.
                                GenericBound::Trait(
                                    _,
                                    TraitBoundModifier::Maybe | TraitBoundModifier::MaybeConstMaybe,
                                ) => None,
                                GenericBound::Outlives(ref lifetime) => {
                                    if lifetime_bound.is_none() {
                                        lifetime_bound = Some(this.lower_lifetime(lifetime));
                                    }
                                    None
                                }
                            },
                        ));
                    let lifetime_bound =
                        lifetime_bound.unwrap_or_else(|| this.elided_dyn_bound(t.span));
                    (bounds, lifetime_bound)
                });
                hir::TyKind::TraitObject(bounds, lifetime_bound, kind)
            }
            TyKind::ImplTrait(def_node_id, ref bounds) => {
                let span = t.span;
                match itctx {
                    ImplTraitContext::ReturnPositionOpaqueTy { origin } => self
                        .lower_opaque_impl_trait(span, origin, def_node_id, |this| {
                            this.lower_param_bounds(bounds, itctx)
                        }),
                    ImplTraitContext::TypeAliasesOpaqueTy => {
                        let nested_itctx = ImplTraitContext::TypeAliasesOpaqueTy;
                        self.lower_opaque_impl_trait(
                            span,
                            hir::OpaqueTyOrigin::TyAlias,
                            def_node_id,
                            |this| this.lower_param_bounds(bounds, nested_itctx),
                        )
                    }
                    ImplTraitContext::Universal(
                        in_band_ty_params,
                        in_band_ty_bounds,
                        parent_def_id,
                    ) => {
                        // Add a definition for the in-band `Param`.
                        let def_id = self.resolver.local_def_id(def_node_id);

                        let hir_bounds = self.lower_param_bounds(
                            bounds,
                            ImplTraitContext::Universal(
                                in_band_ty_params,
                                in_band_ty_bounds,
                                parent_def_id,
                            ),
                        );
                        // Set the name to `impl Bound1 + Bound2`.
                        let ident = Ident::from_str_and_span(&pprust::ty_to_string(t), span);
                        in_band_ty_params.push(hir::GenericParam {
                            hir_id: self.lower_node_id(def_node_id),
                            name: ParamName::Plain(self.lower_ident(ident)),
                            pure_wrt_drop: false,
                            span: self.lower_span(span),
                            kind: hir::GenericParamKind::Type { default: None, synthetic: true },
                            colon_span: None,
                        });
                        if let Some(preds) = self.lower_generic_bound_predicate(
                            ident,
                            def_node_id,
                            &GenericParamKind::Type { default: None },
                            hir_bounds,
                        ) {
                            in_band_ty_bounds.push(preds)
                        }

                        hir::TyKind::Path(hir::QPath::Resolved(
                            None,
                            self.arena.alloc(hir::Path {
                                span: self.lower_span(span),
                                res: Res::Def(DefKind::TyParam, def_id.to_def_id()),
                                segments: arena_vec![self; hir::PathSegment::from_ident(self.lower_ident(ident))],
                            }),
                        ))
                    }
                    ImplTraitContext::Disallowed(position) => {
                        let mut err = struct_span_err!(
                            self.sess,
                            t.span,
                            E0562,
                            "`impl Trait` only allowed in function and inherent method return types, not in {}",
                            position
                        );
                        err.emit();
                        hir::TyKind::Err
                    }
                }
            }
            TyKind::MacCall(_) => panic!("`TyKind::MacCall` should have been expanded by now"),
            TyKind::CVarArgs => {
                self.sess.delay_span_bug(
                    t.span,
                    "`TyKind::CVarArgs` should have been handled elsewhere",
                );
                hir::TyKind::Err
            }
        };

        hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.lower_node_id(t.id) }
    }

    #[tracing::instrument(level = "debug", skip(self, lower_bounds))]
    fn lower_opaque_impl_trait(
        &mut self,
        span: Span,
        origin: hir::OpaqueTyOrigin,
        opaque_ty_node_id: NodeId,
        lower_bounds: impl FnOnce(&mut Self) -> hir::GenericBounds<'hir>,
    ) -> hir::TyKind<'hir> {
        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::OpaqueTy, span, None);

        let opaque_ty_def_id = self.resolver.local_def_id(opaque_ty_node_id);

        let mut collected_lifetimes = FxHashMap::default();
        self.with_hir_id_owner(opaque_ty_node_id, |lctx| {
            let hir_bounds = if origin == hir::OpaqueTyOrigin::TyAlias {
                lower_bounds(lctx)
            } else {
                lctx.while_capturing_lifetimes(
                    opaque_ty_def_id,
                    &mut collected_lifetimes,
                    lower_bounds,
                )
            };
            debug!(?collected_lifetimes);

            let lifetime_defs = lctx.arena.alloc_from_iter(collected_lifetimes.iter().map(
                |(_, &(span, p_id, p_name, _))| {
                    let hir_id = lctx.lower_node_id(p_id);
                    debug_assert_ne!(lctx.resolver.opt_local_def_id(p_id), None);

                    let kind = if p_name.ident().name == kw::UnderscoreLifetime {
                        hir::LifetimeParamKind::Elided
                    } else {
                        hir::LifetimeParamKind::Explicit
                    };

                    hir::GenericParam {
                        hir_id,
                        name: p_name,
                        span,
                        pure_wrt_drop: false,
                        kind: hir::GenericParamKind::Lifetime { kind },
                        colon_span: None,
                    }
                },
            ));

            debug!("lower_opaque_impl_trait: lifetime_defs={:#?}", lifetime_defs);

            let opaque_ty_item = hir::OpaqueTy {
                generics: self.arena.alloc(hir::Generics {
                    params: lifetime_defs,
                    predicates: &[],
                    has_where_clause: false,
                    where_clause_span: lctx.lower_span(span),
                    span: lctx.lower_span(span),
                }),
                bounds: hir_bounds,
                origin,
            };

            trace!("lower_opaque_impl_trait: {:#?}", opaque_ty_def_id);
            lctx.generate_opaque_type(opaque_ty_def_id, opaque_ty_item, span, opaque_ty_span)
        });

        let lifetimes = self.arena.alloc_from_iter(collected_lifetimes.into_iter().map(
            |(_, (span, _, p_name, res))| {
                let id = self.resolver.next_node_id();
                let ident = Ident::new(p_name.ident().name, span);
                let l = self.new_named_lifetime_with_res(id, span, ident, res);
                hir::GenericArg::Lifetime(l)
            },
        ));

        debug!("lower_opaque_impl_trait: lifetimes={:#?}", lifetimes);

        // `impl Trait` now just becomes `Foo<'a, 'b, ..>`.
        hir::TyKind::OpaqueDef(hir::ItemId { def_id: opaque_ty_def_id }, lifetimes)
    }

    /// Registers a new opaque type with the proper `NodeId`s and
    /// returns the lowered node-ID for the opaque type.
    fn generate_opaque_type(
        &mut self,
        opaque_ty_id: LocalDefId,
        opaque_ty_item: hir::OpaqueTy<'hir>,
        span: Span,
        opaque_ty_span: Span,
    ) -> hir::OwnerNode<'hir> {
        let opaque_ty_item_kind = hir::ItemKind::OpaqueTy(opaque_ty_item);
        // Generate an `type Foo = impl Trait;` declaration.
        trace!("registering opaque type with id {:#?}", opaque_ty_id);
        let opaque_ty_item = hir::Item {
            def_id: opaque_ty_id,
            ident: Ident::empty(),
            kind: opaque_ty_item_kind,
            vis_span: self.lower_span(span.shrink_to_lo()),
            span: self.lower_span(opaque_ty_span),
        };
        hir::OwnerNode::Item(self.arena.alloc(opaque_ty_item))
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
    // `fn_def_id`: if `Some`, impl Trait arguments are lowered into generic parameters on the
    //      given DefId, otherwise impl Trait is disallowed. Must be `Some` if
    //      `make_ret_async` is also `Some`.
    // `impl_trait_return_allow`: determines whether `impl Trait` can be used in return position.
    //      This guards against trait declarations and implementations where `impl Trait` is
    //      disallowed.
    // `make_ret_async`: if `Some`, converts `-> T` into `-> impl Future<Output = T>` in the
    //      return type. This is used for `async fn` declarations. The `NodeId` is the ID of the
    //      return type `impl Trait` item.
    fn lower_fn_decl(
        &mut self,
        decl: &FnDecl,
        mut in_band_ty_params: Option<(
            NodeId,
            &mut Vec<hir::GenericParam<'hir>>,
            &mut Vec<hir::WherePredicate<'hir>>,
        )>,
        kind: FnDeclKind,
        make_ret_async: Option<NodeId>,
    ) -> &'hir hir::FnDecl<'hir> {
        debug!(
            "lower_fn_decl(\
            fn_decl: {:?}, \
            in_band_ty_params: {:?}, \
            kind: {:?}, \
            make_ret_async: {:?})",
            decl, in_band_ty_params, kind, make_ret_async,
        );

        let c_variadic = decl.c_variadic();

        // Skip the `...` (`CVarArgs`) trailing arguments from the AST,
        // as they are not explicit in HIR/Ty function signatures.
        // (instead, the `c_variadic` flag is set to `true`)
        let mut inputs = &decl.inputs[..];
        if c_variadic {
            inputs = &inputs[..inputs.len() - 1];
        }
        let inputs = self.arena.alloc_from_iter(inputs.iter().map(|param| {
            if let Some((_, ibty, ibpb)) = &mut in_band_ty_params {
                self.lower_ty_direct(
                    &param.ty,
                    ImplTraitContext::Universal(ibty, ibpb, self.current_hir_id_owner),
                )
            } else {
                self.lower_ty_direct(
                    &param.ty,
                    ImplTraitContext::Disallowed(match kind {
                        FnDeclKind::Fn | FnDeclKind::Inherent => {
                            unreachable!("fn should allow in-band lifetimes")
                        }
                        FnDeclKind::ExternFn => ImplTraitPosition::ExternFnParam,
                        FnDeclKind::Closure => ImplTraitPosition::ClosureParam,
                        FnDeclKind::Pointer => ImplTraitPosition::PointerParam,
                        FnDeclKind::Trait => ImplTraitPosition::TraitParam,
                        FnDeclKind::Impl => ImplTraitPosition::ImplParam,
                    }),
                )
            }
        }));

        let output = if let Some(ret_id) = make_ret_async {
            self.lower_async_fn_ret_ty(
                &decl.output,
                in_band_ty_params.expect("`make_ret_async` but no `fn_def_id`").0,
                ret_id,
            )
        } else {
            match decl.output {
                FnRetTy::Ty(ref ty) => {
                    let context = match in_band_ty_params {
                        Some((node_id, _, _)) if kind.impl_trait_return_allowed() => {
                            let fn_def_id = self.resolver.local_def_id(node_id);
                            ImplTraitContext::ReturnPositionOpaqueTy {
                                origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                            }
                        }
                        _ => ImplTraitContext::Disallowed(match kind {
                            FnDeclKind::Fn | FnDeclKind::Inherent => {
                                unreachable!("fn should allow in-band lifetimes")
                            }
                            FnDeclKind::ExternFn => ImplTraitPosition::ExternFnReturn,
                            FnDeclKind::Closure => ImplTraitPosition::ClosureReturn,
                            FnDeclKind::Pointer => ImplTraitPosition::PointerReturn,
                            FnDeclKind::Trait => ImplTraitPosition::TraitReturn,
                            FnDeclKind::Impl => ImplTraitPosition::ImplReturn,
                        }),
                    };
                    hir::FnRetTy::Return(self.lower_ty(ty, context))
                }
                FnRetTy::Default(span) => hir::FnRetTy::DefaultReturn(self.lower_span(span)),
            }
        };

        self.arena.alloc(hir::FnDecl {
            inputs,
            output,
            c_variadic,
            implicit_self: decl.inputs.get(0).map_or(hir::ImplicitSelfKind::None, |arg| {
                use BindingMode::{ByRef, ByValue};
                let is_mutable_pat = matches!(
                    arg.pat.kind,
                    PatKind::Ident(ByValue(Mutability::Mut) | ByRef(Mutability::Mut), ..)
                );

                match arg.ty.kind {
                    TyKind::ImplicitSelf if is_mutable_pat => hir::ImplicitSelfKind::Mut,
                    TyKind::ImplicitSelf => hir::ImplicitSelfKind::Imm,
                    // Given we are only considering `ImplicitSelf` types, we needn't consider
                    // the case where we have a mutable pattern to a reference as that would
                    // no longer be an `ImplicitSelf`.
                    TyKind::Rptr(_, ref mt)
                        if mt.ty.kind.is_implicit_self() && mt.mutbl == ast::Mutability::Mut =>
                    {
                        hir::ImplicitSelfKind::MutRef
                    }
                    TyKind::Rptr(_, ref mt) if mt.ty.kind.is_implicit_self() => {
                        hir::ImplicitSelfKind::ImmRef
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
    // `inputs`: lowered types of parameters to the function (used to collect lifetimes)
    // `output`: unlowered output type (`T` in `-> T`)
    // `fn_def_id`: `DefId` of the parent function (used to create child impl trait definition)
    // `opaque_ty_node_id`: `NodeId` of the opaque `impl Trait` type that should be created
    // `elided_lt_replacement`: replacement for elided lifetimes in the return type
    #[tracing::instrument(level = "debug", skip(self))]
    fn lower_async_fn_ret_ty(
        &mut self,
        output: &FnRetTy,
        fn_node_id: NodeId,
        opaque_ty_node_id: NodeId,
    ) -> hir::FnRetTy<'hir> {
        let span = output.span();

        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::Async, span, None);

        let opaque_ty_def_id = self.resolver.local_def_id(opaque_ty_node_id);
        let fn_def_id = self.resolver.local_def_id(fn_node_id);

        // When we create the opaque type for this async fn, it is going to have
        // to capture all the lifetimes involved in the signature (including in the
        // return type). This is done by introducing lifetime parameters for:
        //
        // - all the explicitly declared lifetimes from the impl and function itself;
        // - all the elided lifetimes in the fn arguments;
        // - all the elided lifetimes in the return type.
        //
        // So for example in this snippet:
        //
        // ```rust
        // impl<'a> Foo<'a> {
        //   async fn bar<'b>(&self, x: &'b Vec<f64>, y: &str) -> &u32 {
        //   //               ^ '0                       ^ '1     ^ '2
        //   // elided lifetimes used below
        //   }
        // }
        // ```
        //
        // we would create an opaque type like:
        //
        // ```
        // type Bar<'a, 'b, '0, '1, '2> = impl Future<Output = &'2 u32>;
        // ```
        //
        // and we would then desugar `bar` to the equivalent of:
        //
        // ```rust
        // impl<'a> Foo<'a> {
        //   fn bar<'b, '0, '1>(&'0 self, x: &'b Vec<f64>, y: &'1 str) -> Bar<'a, 'b, '0, '1, '_>
        // }
        // ```
        //
        // Note that the final parameter to `Bar` is `'_`, not `'2` --
        // this is because the elided lifetimes from the return type
        // should be figured out using the ordinary elision rules, and
        // this desugaring achieves that.

        // Calculate all the lifetimes that should be captured
        // by the opaque type. This should include all in-scope
        // lifetime parameters, including those defined in-band.

        let mut captures = FxHashMap::default();

        let extra_lifetime_params = self.resolver.take_extra_lifetime_params(opaque_ty_node_id);
        debug!(?extra_lifetime_params);
        for (ident, outer_node_id, outer_res) in extra_lifetime_params {
            let Ident { name, span } = ident;
            let outer_def_id = self.resolver.local_def_id(outer_node_id);
            let inner_node_id = self.resolver.next_node_id();

            // Add a definition for the in scope lifetime def.
            self.resolver.create_def(
                opaque_ty_def_id,
                inner_node_id,
                DefPathData::LifetimeNs(name),
                ExpnId::root(),
                span.with_parent(None),
            );

            let (p_name, inner_res) = match outer_res {
                // Input lifetime like `'a`:
                LifetimeRes::Param { param, .. } => {
                    (hir::ParamName::Plain(ident), LifetimeRes::Param { param, binder: fn_node_id })
                }
                // Input lifetime like `'1`:
                LifetimeRes::Fresh { param, .. } => (
                    hir::ParamName::Fresh(outer_def_id),
                    LifetimeRes::Fresh { param, binder: fn_node_id },
                ),
                LifetimeRes::Static | LifetimeRes::Error => continue,
                res => {
                    panic!("Unexpected lifetime resolution {:?} for {:?} at {:?}", res, ident, span)
                }
            };

            captures.insert(outer_def_id, (span, inner_node_id, p_name, inner_res));
        }

        debug!(?captures);

        self.with_hir_id_owner(opaque_ty_node_id, |this| {
            let future_bound =
                this.while_capturing_lifetimes(opaque_ty_def_id, &mut captures, |this| {
                    // We have to be careful to get elision right here. The
                    // idea is that we create a lifetime parameter for each
                    // lifetime in the return type.  So, given a return type
                    // like `async fn foo(..) -> &[&u32]`, we lower to `impl
                    // Future<Output = &'1 [ &'2 u32 ]>`.
                    //
                    // Then, we will create `fn foo(..) -> Foo<'_, '_>`, and
                    // hence the elision takes place at the fn site.
                    this.lower_async_fn_output_type_to_future_bound(output, fn_def_id, span)
                });
            debug!("lower_async_fn_ret_ty: future_bound={:#?}", future_bound);
            debug!("lower_async_fn_ret_ty: captures={:#?}", captures);

            let generic_params =
                this.arena.alloc_from_iter(captures.iter().map(|(_, &(span, p_id, p_name, _))| {
                    let hir_id = this.lower_node_id(p_id);
                    debug_assert_ne!(this.resolver.opt_local_def_id(p_id), None);

                    let kind = if p_name.ident().name == kw::UnderscoreLifetime {
                        hir::LifetimeParamKind::Elided
                    } else {
                        hir::LifetimeParamKind::Explicit
                    };

                    hir::GenericParam {
                        hir_id,
                        name: p_name,
                        span,
                        pure_wrt_drop: false,
                        kind: hir::GenericParamKind::Lifetime { kind },
                        colon_span: None,
                    }
                }));
            debug!("lower_async_fn_ret_ty: generic_params={:#?}", generic_params);

            let opaque_ty_item = hir::OpaqueTy {
                generics: this.arena.alloc(hir::Generics {
                    params: generic_params,
                    predicates: &[],
                    has_where_clause: false,
                    where_clause_span: this.lower_span(span),
                    span: this.lower_span(span),
                }),
                bounds: arena_vec![this; future_bound],
                origin: hir::OpaqueTyOrigin::AsyncFn(fn_def_id),
            };

            trace!("exist ty from async fn def id: {:#?}", opaque_ty_def_id);
            this.generate_opaque_type(opaque_ty_def_id, opaque_ty_item, span, opaque_ty_span)
        });

        // As documented above, we need to create the lifetime
        // arguments to our opaque type. Continuing with our example,
        // we're creating the type arguments for the return type:
        //
        // ```
        // Bar<'a, 'b, '0, '1, '_>
        // ```
        //
        // For the "input" lifetime parameters, we wish to create
        // references to the parameters themselves, including the
        // "implicit" ones created from parameter types (`'a`, `'b`,
        // '`0`, `'1`).
        //
        // For the "output" lifetime parameters, we just want to
        // generate `'_`.
        let generic_args =
            self.arena.alloc_from_iter(captures.into_iter().map(|(_, (span, _, p_name, res))| {
                let id = self.resolver.next_node_id();
                let ident = Ident::new(p_name.ident().name, span);
                let l = self.new_named_lifetime_with_res(id, span, ident, res);
                hir::GenericArg::Lifetime(l)
            }));

        // Create the `Foo<...>` reference itself. Note that the `type
        // Foo = impl Trait` is, internally, created as a child of the
        // async fn, so the *type parameters* are inherited.  It's
        // only the lifetime parameters that we must supply.
        let opaque_ty_ref =
            hir::TyKind::OpaqueDef(hir::ItemId { def_id: opaque_ty_def_id }, generic_args);
        let opaque_ty = self.ty(opaque_ty_span, opaque_ty_ref);
        hir::FnRetTy::Return(self.arena.alloc(opaque_ty))
    }

    /// Transforms `-> T` into `Future<Output = T>`.
    fn lower_async_fn_output_type_to_future_bound(
        &mut self,
        output: &FnRetTy,
        fn_def_id: LocalDefId,
        span: Span,
    ) -> hir::GenericBound<'hir> {
        // Compute the `T` in `Future<Output = T>` from the return type.
        let output_ty = match output {
            FnRetTy::Ty(ty) => {
                // Not `OpaqueTyOrigin::AsyncFn`: that's only used for the
                // `impl Future` opaque type that `async fn` implicitly
                // generates.
                let context = ImplTraitContext::ReturnPositionOpaqueTy {
                    origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                };
                self.lower_ty(ty, context)
            }
            FnRetTy::Default(ret_ty_span) => self.arena.alloc(self.ty_tup(*ret_ty_span, &[])),
        };

        // "<Output = T>"
        let future_args = self.arena.alloc(hir::GenericArgs {
            args: &[],
            bindings: arena_vec![self; self.output_ty_binding(span, output_ty)],
            parenthesized: false,
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

    fn lower_param_bound(
        &mut self,
        tpb: &GenericBound,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericBound<'hir> {
        match tpb {
            GenericBound::Trait(p, modifier) => hir::GenericBound::Trait(
                self.lower_poly_trait_ref(p, itctx),
                self.lower_trait_bound_modifier(*modifier),
            ),
            GenericBound::Outlives(lifetime) => {
                hir::GenericBound::Outlives(self.lower_lifetime(lifetime))
            }
        }
    }

    fn lower_lifetime(&mut self, l: &Lifetime) -> hir::Lifetime {
        let span = self.lower_span(l.ident.span);
        let ident = self.lower_ident(l.ident);
        let res = self
            .resolver
            .get_lifetime_res(l.id)
            .unwrap_or_else(|| panic!("Missing resolution for lifetime {:?} at {:?}", l, span));
        self.new_named_lifetime_with_res(l.id, span, ident, res)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn new_named_lifetime_with_res(
        &mut self,
        id: NodeId,
        span: Span,
        ident: Ident,
        res: LifetimeRes,
    ) -> hir::Lifetime {
        debug!(?self.captured_lifetimes);
        let name = match res {
            LifetimeRes::Param { param, binder } => {
                debug_assert_ne!(ident.name, kw::UnderscoreLifetime);
                let p_name = ParamName::Plain(ident);
                if let Some(LifetimeCaptureContext { parent_def_id, captures, binders_to_ignore }) =
                    &mut self.captured_lifetimes
                    && !binders_to_ignore.contains(&binder)
                {
                    match captures.entry(param) {
                        Entry::Occupied(_) => {}
                        Entry::Vacant(v) => {
                            let p_id = self.resolver.next_node_id();
                            self.resolver.create_def(
                                *parent_def_id,
                                p_id,
                                DefPathData::LifetimeNs(p_name.ident().name),
                                ExpnId::root(),
                                span.with_parent(None),
                            );

                            v.insert((span, p_id, p_name, res));
                        }
                    }
                }
                hir::LifetimeName::Param(p_name)
            }
            LifetimeRes::Fresh { mut param, binder } => {
                debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
                if let Some(LifetimeCaptureContext { parent_def_id, captures, binders_to_ignore }) =
                    &mut self.captured_lifetimes
                    && !binders_to_ignore.contains(&binder)
                {
                    match captures.entry(param) {
                        Entry::Occupied(o) => param = self.resolver.local_def_id(o.get().1),
                        Entry::Vacant(v) => {
                            let p_id = self.resolver.next_node_id();
                            let p_def_id = self.resolver.create_def(
                                *parent_def_id,
                                p_id,
                                DefPathData::LifetimeNs(kw::UnderscoreLifetime),
                                ExpnId::root(),
                                span.with_parent(None),
                            );

                            let p_name = ParamName::Fresh(param);
                            v.insert((span, p_id, p_name, res));
                            param = p_def_id;
                        }
                    }
                }
                let p_name = ParamName::Fresh(param);
                hir::LifetimeName::Param(p_name)
            }
            LifetimeRes::Anonymous { binder, elided } => {
                let l_name = if elided {
                    hir::LifetimeName::Implicit
                } else {
                    hir::LifetimeName::Underscore
                };
                if let Some(LifetimeCaptureContext { parent_def_id, captures, binders_to_ignore }) =
                    &mut self.captured_lifetimes
                    && !binders_to_ignore.contains(&binder)
                {
                    let p_id = self.resolver.next_node_id();
                    let p_def_id = self.resolver.create_def(
                        *parent_def_id,
                        p_id,
                        DefPathData::LifetimeNs(kw::UnderscoreLifetime),
                        ExpnId::root(),
                        span.with_parent(None),
                    );
                    let p_name = ParamName::Fresh(p_def_id);
                    captures.insert(p_def_id, (span, p_id, p_name, res));
                    hir::LifetimeName::Param(p_name)
                } else {
                    l_name
                }
            }
            LifetimeRes::Static => hir::LifetimeName::Static,
            LifetimeRes::Error => hir::LifetimeName::Error,
            res => panic!("Unexpected lifetime resolution {:?} for {:?} at {:?}", res, ident, span),
        };
        debug!(?self.captured_lifetimes);
        hir::Lifetime { hir_id: self.lower_node_id(id), span: self.lower_span(span), name }
    }

    fn lower_generic_params_mut<'s>(
        &'s mut self,
        params: &'s [GenericParam],
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> + Captures<'a> + Captures<'s> {
        params.iter().map(move |param| self.lower_generic_param(param))
    }

    fn lower_generic_params(&mut self, params: &[GenericParam]) -> &'hir [hir::GenericParam<'hir>] {
        self.arena.alloc_from_iter(self.lower_generic_params_mut(params))
    }

    fn lower_generic_param(&mut self, param: &GenericParam) -> hir::GenericParam<'hir> {
        let (name, kind) = match param.kind {
            GenericParamKind::Lifetime => {
                let param_name = if param.ident.name == kw::StaticLifetime
                    || param.ident.name == kw::UnderscoreLifetime
                {
                    ParamName::Error
                } else {
                    let ident = self.lower_ident(param.ident);
                    ParamName::Plain(ident)
                };
                let kind =
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit };

                (param_name, kind)
            }
            GenericParamKind::Type { ref default, .. } => {
                let kind = hir::GenericParamKind::Type {
                    default: default.as_ref().map(|x| {
                        self.lower_ty(x, ImplTraitContext::Disallowed(ImplTraitPosition::Type))
                    }),
                    synthetic: false,
                };

                (hir::ParamName::Plain(self.lower_ident(param.ident)), kind)
            }
            GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                let ty = self.lower_ty(&ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type));
                let default = default.as_ref().map(|def| self.lower_anon_const(def));
                (
                    hir::ParamName::Plain(self.lower_ident(param.ident)),
                    hir::GenericParamKind::Const { ty, default },
                )
            }
        };
        let name = match name {
            hir::ParamName::Plain(ident) => hir::ParamName::Plain(self.lower_ident(ident)),
            name => name,
        };

        let hir_id = self.lower_node_id(param.id);
        self.lower_attrs(hir_id, &param.attrs);
        hir::GenericParam {
            hir_id,
            name,
            span: self.lower_span(param.span()),
            pure_wrt_drop: self.sess.contains_name(&param.attrs, sym::may_dangle),
            kind,
            colon_span: param.colon_span.map(|s| self.lower_span(s)),
        }
    }

    fn lower_trait_ref(
        &mut self,
        p: &TraitRef,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::TraitRef<'hir> {
        let path = match self.lower_qpath(p.ref_id, &None, &p.path, ParamMode::Explicit, itctx) {
            hir::QPath::Resolved(None, path) => path,
            qpath => panic!("lower_trait_ref: unexpected QPath `{:?}`", qpath),
        };
        hir::TraitRef { path, hir_ref_id: self.lower_node_id(p.ref_id) }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn lower_poly_trait_ref(
        &mut self,
        p: &PolyTraitRef,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::PolyTraitRef<'hir> {
        let bound_generic_params = self.lower_generic_params(&p.bound_generic_params);

        let trait_ref = self.with_lifetime_binder(p.trait_ref.ref_id, |this| {
            this.lower_trait_ref(&p.trait_ref, itctx.reborrow())
        });

        hir::PolyTraitRef { bound_generic_params, trait_ref, span: self.lower_span(p.span) }
    }

    fn lower_mt(&mut self, mt: &MutTy, itctx: ImplTraitContext<'_, 'hir>) -> hir::MutTy<'hir> {
        hir::MutTy { ty: self.lower_ty(&mt.ty, itctx), mutbl: mt.mutbl }
    }

    fn lower_param_bounds(
        &mut self,
        bounds: &[GenericBound],
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericBounds<'hir> {
        self.arena.alloc_from_iter(self.lower_param_bounds_mut(bounds, itctx))
    }

    fn lower_param_bounds_mut<'s>(
        &'s mut self,
        bounds: &'s [GenericBound],
        mut itctx: ImplTraitContext<'s, 'hir>,
    ) -> impl Iterator<Item = hir::GenericBound<'hir>> + Captures<'s> + Captures<'a> {
        bounds.iter().map(move |bound| self.lower_param_bound(bound, itctx.reborrow()))
    }

    /// Lowers a block directly to an expression, presuming that it
    /// has no attributes and is not targeted by a `break`.
    fn lower_block_expr(&mut self, b: &Block) -> hir::Expr<'hir> {
        let block = self.lower_block(b, false);
        self.expr_block(block, AttrVec::new())
    }

    fn lower_array_length(&mut self, c: &AnonConst) -> hir::ArrayLen {
        match c.value.kind {
            ExprKind::Underscore => {
                if self.sess.features_untracked().generic_arg_infer {
                    hir::ArrayLen::Infer(self.lower_node_id(c.id), c.value.span)
                } else {
                    feature_err(
                        &self.sess.parse_sess,
                        sym::generic_arg_infer,
                        c.value.span,
                        "using `_` for array lengths is unstable",
                    )
                    .emit();
                    hir::ArrayLen::Body(self.lower_anon_const(c))
                }
            }
            _ => hir::ArrayLen::Body(self.lower_anon_const(c)),
        }
    }

    fn lower_anon_const(&mut self, c: &AnonConst) -> hir::AnonConst {
        self.with_new_scopes(|this| hir::AnonConst {
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

            // `MaybeConstMaybe` will cause an error during AST validation, but we need to pick a
            // placeholder for compilation to proceed.
            TraitBoundModifier::MaybeConstMaybe | TraitBoundModifier::Maybe => {
                hir::TraitBoundModifier::Maybe
            }
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
        let local = hir::Local { hir_id, init, pat, source, span: self.lower_span(span), ty: None };
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
        self.pat_ident_binding_mode(span, ident, hir::BindingAnnotation::Unannotated)
    }

    fn pat_ident_mut(&mut self, span: Span, ident: Ident) -> (hir::Pat<'hir>, hir::HirId) {
        self.pat_ident_binding_mode_mut(span, ident, hir::BindingAnnotation::Unannotated)
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
    fn elided_dyn_bound(&mut self, span: Span) -> hir::Lifetime {
        let r = hir::Lifetime {
            hir_id: self.next_id(),
            span: self.lower_span(span),
            name: hir::LifetimeName::ImplicitObjectLifetimeDefault,
        };
        debug!("elided_dyn_bound: r={:?}", r);
        r
    }
}

/// Helper struct for delayed construction of GenericArgs.
struct GenericArgsCtor<'hir> {
    args: SmallVec<[hir::GenericArg<'hir>; 4]>,
    bindings: &'hir [hir::TypeBinding<'hir>],
    parenthesized: bool,
    span: Span,
}

impl<'hir> GenericArgsCtor<'hir> {
    fn is_empty(&self) -> bool {
        self.args.is_empty() && self.bindings.is_empty() && !self.parenthesized
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
