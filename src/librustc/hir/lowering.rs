// ignore-tidy-filelength

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
//! 'folding' an existing one), then you create a new ID using `next_id()`.
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
//! All nodes should have real spans, don't use dummy spans. Tools are likely to
//! get confused if the spans from leaf AST nodes occur in multiple places
//! in the HIR, especially for multiple identifiers.

mod expr;
mod item;

use crate::dep_graph::DepGraph;
use crate::hir::{self, ParamName};
use crate::hir::HirVec;
use crate::hir::map::{DefKey, DefPathData, Definitions};
use crate::hir::def_id::{DefId, DefIndex, CRATE_DEF_INDEX};
use crate::hir::def::{Namespace, Res, DefKind, PartialRes, PerNS};
use crate::hir::{GenericArg, ConstArg};
use crate::hir::ptr::P;
use crate::lint::builtin::{self, PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES,
                    ELIDED_LIFETIMES_IN_PATHS};
use crate::middle::cstore::CrateStore;
use crate::session::Session;
use crate::session::config::nightly_options;
use crate::util::common::FN_OUTPUT_NAME;
use crate::util::nodemap::{DefIdMap, NodeMap};
use errors::Applicability;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::sync::Lrc;

use std::collections::BTreeMap;
use std::mem;
use smallvec::SmallVec;
use syntax::attr;
use syntax::ast;
use syntax::ptr::P as AstP;
use syntax::ast::*;
use syntax::errors;
use syntax::ext::base::SpecialDerives;
use syntax::ext::hygiene::ExpnId;
use syntax::print::pprust;
use syntax::source_map::{respan, ExpnData, ExpnKind, DesugaringKind, Spanned};
use syntax::symbol::{kw, sym, Symbol};
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax::parse::token::{self, Token};
use syntax::visit::{self, Visitor};
use syntax_pos::Span;

const HIR_ID_COUNTER_LOCKED: u32 = 0xFFFFFFFF;

pub struct LoweringContext<'a> {
    crate_root: Option<Symbol>,

    /// Used to assign IDs to HIR nodes that do not directly correspond to AST nodes.
    sess: &'a Session,

    cstore: &'a dyn CrateStore,

    resolver: &'a mut dyn Resolver,

    /// The items being lowered are collected here.
    items: BTreeMap<hir::HirId, hir::Item>,

    trait_items: BTreeMap<hir::TraitItemId, hir::TraitItem>,
    impl_items: BTreeMap<hir::ImplItemId, hir::ImplItem>,
    bodies: BTreeMap<hir::BodyId, hir::Body>,
    exported_macros: Vec<hir::MacroDef>,
    non_exported_macro_attrs: Vec<ast::Attribute>,

    trait_impls: BTreeMap<DefId, Vec<hir::HirId>>,

    modules: BTreeMap<NodeId, hir::ModuleItems>,

    generator_kind: Option<hir::GeneratorKind>,

    /// Used to get the current `fn`'s def span to point to when using `await`
    /// outside of an `async fn`.
    current_item: Option<Span>,

    catch_scopes: Vec<NodeId>,
    loop_scopes: Vec<NodeId>,
    is_in_loop_condition: bool,
    is_in_trait_impl: bool,
    is_in_dyn_type: bool,

    /// What to do when we encounter either an "anonymous lifetime
    /// reference". The term "anonymous" is meant to encompass both
    /// `'_` lifetimes as well as fully elided cases where nothing is
    /// written at all (e.g., `&T` or `std::cell::Ref<T>`).
    anonymous_lifetime_mode: AnonymousLifetimeMode,

    /// Used to create lifetime definitions from in-band lifetime usages.
    /// e.g., `fn foo(x: &'x u8) -> &'x u8` to `fn foo<'x>(x: &'x u8) -> &'x u8`
    /// When a named lifetime is encountered in a function or impl header and
    /// has not been defined
    /// (i.e., it doesn't appear in the in_scope_lifetimes list), it is added
    /// to this list. The results of this list are then added to the list of
    /// lifetime definitions in the corresponding impl or function generics.
    lifetimes_to_define: Vec<(Span, ParamName)>,

    /// `true` if in-band lifetimes are being collected. This is used to
    /// indicate whether or not we're in a place where new lifetimes will result
    /// in in-band lifetime definitions, such a function or an impl header,
    /// including implicit lifetimes from `impl_header_lifetime_elision`.
    is_collecting_in_band_lifetimes: bool,

    /// Currently in-scope lifetimes defined in impl headers, fn headers, or HRTB.
    /// When `is_collectin_in_band_lifetimes` is true, each lifetime is checked
    /// against this list to see if it is already in-scope, or if a definition
    /// needs to be created for it.
    ///
    /// We always store a `modern()` version of the param-name in this
    /// vector.
    in_scope_lifetimes: Vec<ParamName>,

    current_module: NodeId,

    type_def_lifetime_params: DefIdMap<usize>,

    current_hir_id_owner: Vec<(DefIndex, u32)>,
    item_local_id_counters: NodeMap<u32>,
    node_id_to_hir_id: IndexVec<NodeId, hir::HirId>,

    allow_try_trait: Option<Lrc<[Symbol]>>,
    allow_gen_future: Option<Lrc<[Symbol]>>,
}

pub trait Resolver {
    /// Obtains resolution for a `NodeId` with a single resolution.
    fn get_partial_res(&mut self, id: NodeId) -> Option<PartialRes>;

    /// Obtains per-namespace resolutions for `use` statement with the given `NodeId`.
    fn get_import_res(&mut self, id: NodeId) -> PerNS<Option<Res<NodeId>>>;

    /// Obtains resolution for a label with the given `NodeId`.
    fn get_label_res(&mut self, id: NodeId) -> Option<NodeId>;

    /// We must keep the set of definitions up to date as we add nodes that weren't in the AST.
    /// This should only return `None` during testing.
    fn definitions(&mut self) -> &mut Definitions;

    /// Given suffix `["b", "c", "d"]`, creates an AST path for `[::crate_root]::b::c::d` and
    /// resolves it based on `is_value`.
    fn resolve_str_path(
        &mut self,
        span: Span,
        crate_root: Option<Symbol>,
        components: &[Symbol],
        ns: Namespace,
    ) -> (ast::Path, Res<NodeId>);

    fn has_derives(&self, node_id: NodeId, derives: SpecialDerives) -> bool;
}

/// Context of `impl Trait` in code, which determines whether it is allowed in an HIR subtree,
/// and if so, what meaning it has.
#[derive(Debug)]
enum ImplTraitContext<'a> {
    /// Treat `impl Trait` as shorthand for a new universal generic parameter.
    /// Example: `fn foo(x: impl Debug)`, where `impl Debug` is conceptually
    /// equivalent to a fresh universal parameter like `fn foo<T: Debug>(x: T)`.
    ///
    /// Newly generated parameters should be inserted into the given `Vec`.
    Universal(&'a mut Vec<hir::GenericParam>),

    /// Treat `impl Trait` as shorthand for a new opaque type.
    /// Example: `fn foo() -> impl Debug`, where `impl Debug` is conceptually
    /// equivalent to a new opaque type like `type T = impl Debug; fn foo() -> T`.
    ///
    /// We optionally store a `DefId` for the parent item here so we can look up necessary
    /// information later. It is `None` when no information about the context should be stored
    /// (e.g., for consts and statics).
    OpaqueTy(Option<DefId> /* fn def-ID */),

    /// `impl Trait` is not accepted in this position.
    Disallowed(ImplTraitPosition),
}

/// Position in which `impl Trait` is disallowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitPosition {
    /// Disallowed in `let` / `const` / `static` bindings.
    Binding,

    /// All other posiitons.
    Other,
}

impl<'a> ImplTraitContext<'a> {
    #[inline]
    fn disallowed() -> Self {
        ImplTraitContext::Disallowed(ImplTraitPosition::Other)
    }

    fn reborrow(&'b mut self) -> ImplTraitContext<'b> {
        use self::ImplTraitContext::*;
        match self {
            Universal(params) => Universal(params),
            OpaqueTy(fn_def_id) => OpaqueTy(*fn_def_id),
            Disallowed(pos) => Disallowed(*pos),
        }
    }
}

pub fn lower_crate(
    sess: &Session,
    cstore: &dyn CrateStore,
    dep_graph: &DepGraph,
    krate: &Crate,
    resolver: &mut dyn Resolver,
) -> hir::Crate {
    // We're constructing the HIR here; we don't care what we will
    // read, since we haven't even constructed the *input* to
    // incr. comp. yet.
    dep_graph.assert_ignored();

    LoweringContext {
        crate_root: sess.parse_sess.injected_crate_name.try_get().copied(),
        sess,
        cstore,
        resolver,
        items: BTreeMap::new(),
        trait_items: BTreeMap::new(),
        impl_items: BTreeMap::new(),
        bodies: BTreeMap::new(),
        trait_impls: BTreeMap::new(),
        modules: BTreeMap::new(),
        exported_macros: Vec::new(),
        non_exported_macro_attrs: Vec::new(),
        catch_scopes: Vec::new(),
        loop_scopes: Vec::new(),
        is_in_loop_condition: false,
        is_in_trait_impl: false,
        is_in_dyn_type: false,
        anonymous_lifetime_mode: AnonymousLifetimeMode::PassThrough,
        type_def_lifetime_params: Default::default(),
        current_module: CRATE_NODE_ID,
        current_hir_id_owner: vec![(CRATE_DEF_INDEX, 0)],
        item_local_id_counters: Default::default(),
        node_id_to_hir_id: IndexVec::new(),
        generator_kind: None,
        current_item: None,
        lifetimes_to_define: Vec::new(),
        is_collecting_in_band_lifetimes: false,
        in_scope_lifetimes: Vec::new(),
        allow_try_trait: Some([sym::try_trait][..].into()),
        allow_gen_future: Some([sym::gen_future][..].into()),
    }.lower_crate(krate)
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
    Warn,
    Err,
}

/// What to do when we encounter an **anonymous** lifetime
/// reference. Anonymous lifetime references come in two flavors. You
/// have implicit, or fully elided, references to lifetimes, like the
/// one in `&T` or `Ref<T>`, and you have `'_` lifetimes, like `&'_ T`
/// or `Ref<'_, T>`. These often behave the same, but not always:
///
/// - certain usages of implicit references are deprecated, like
///   `Ref<T>`, and we sometimes just give hard errors in those cases
///   as well.
/// - for object bounds there is a difference: `Box<dyn Foo>` is not
///   the same as `Box<dyn Foo + '_>`.
///
/// We describe the effects of the various modes in terms of three cases:
///
/// - **Modern** -- includes all uses of `'_`, but also the lifetime arg
///   of a `&` (e.g., the missing lifetime in something like `&T`)
/// - **Dyn Bound** -- if you have something like `Box<dyn Foo>`,
///   there is an elided lifetime bound (`Box<dyn Foo + 'X>`). These
///   elided bounds follow special rules. Note that this only covers
///   cases where *nothing* is written; the `'_` in `Box<dyn Foo +
///   '_>` is a case of "modern" elision.
/// - **Deprecated** -- this coverse cases like `Ref<T>`, where the lifetime
///   parameter to ref is completely elided. `Ref<'_, T>` would be the modern,
///   non-deprecated equivalent.
///
/// Currently, the handling of lifetime elision is somewhat spread out
/// between HIR lowering and -- as described below -- the
/// `resolve_lifetime` module. Often we "fallthrough" to that code by generating
/// an "elided" or "underscore" lifetime name. In the future, we probably want to move
/// everything into HIR lowering.
#[derive(Copy, Clone, Debug)]
enum AnonymousLifetimeMode {
    /// For **Modern** cases, create a new anonymous region parameter
    /// and reference that.
    ///
    /// For **Dyn Bound** cases, pass responsibility to
    /// `resolve_lifetime` code.
    ///
    /// For **Deprecated** cases, report an error.
    CreateParameter,

    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    ReportError,

    /// Pass responsibility to `resolve_lifetime` code for all cases.
    PassThrough,
}

struct ImplTraitTypeIdVisitor<'a> { ids: &'a mut SmallVec<[NodeId; 1]> }

impl<'a, 'b> Visitor<'a> for ImplTraitTypeIdVisitor<'b> {
    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            | TyKind::Typeof(_)
            | TyKind::BareFn(_)
            => return,

            TyKind::ImplTrait(id, _) => self.ids.push(id),
            _ => {},
        }
        visit::walk_ty(self, ty);
    }

    fn visit_path_segment(
        &mut self,
        path_span: Span,
        path_segment: &'v PathSegment,
    ) {
        if let Some(ref p) = path_segment.args {
            if let GenericArgs::Parenthesized(_) = **p {
                return;
            }
        }
        visit::walk_path_segment(self, path_span, path_segment)
    }
}

impl<'a> LoweringContext<'a> {
    fn lower_crate(mut self, c: &Crate) -> hir::Crate {
        /// Full-crate AST visitor that inserts into a fresh
        /// `LoweringContext` any information that may be
        /// needed from arbitrary locations in the crate,
        /// e.g., the number of lifetime generic parameters
        /// declared for every type and trait definition.
        struct MiscCollector<'tcx, 'interner> {
            lctx: &'tcx mut LoweringContext<'interner>,
            hir_id_owner: Option<NodeId>,
        }

        impl MiscCollector<'_, '_> {
            fn allocate_use_tree_hir_id_counters(
                &mut self,
                tree: &UseTree,
                owner: DefIndex,
            ) {
                match tree.kind {
                    UseTreeKind::Simple(_, id1, id2) => {
                        for &id in &[id1, id2] {
                            self.lctx.resolver.definitions().create_def_with_parent(
                                owner,
                                id,
                                DefPathData::Misc,
                                ExpnId::root(),
                                tree.prefix.span,
                            );
                            self.lctx.allocate_hir_id_counter(id);
                        }
                    }
                    UseTreeKind::Glob => (),
                    UseTreeKind::Nested(ref trees) => {
                        for &(ref use_tree, id) in trees {
                            let hir_id = self.lctx.allocate_hir_id_counter(id);
                            self.allocate_use_tree_hir_id_counters(use_tree, hir_id.owner);
                        }
                    }
                }
            }

            fn with_hir_id_owner<F, T>(&mut self, owner: Option<NodeId>, f: F) -> T
            where
                F: FnOnce(&mut Self) -> T,
            {
                let old = mem::replace(&mut self.hir_id_owner, owner);
                let r = f(self);
                self.hir_id_owner = old;
                r
            }
        }

        impl<'tcx, 'interner> Visitor<'tcx> for MiscCollector<'tcx, 'interner> {
            fn visit_pat(&mut self, p: &'tcx Pat) {
                if let PatKind::Paren(..) | PatKind::Rest = p.node {
                    // Doesn't generate a HIR node
                } else if let Some(owner) = self.hir_id_owner {
                    self.lctx.lower_node_id_with_owner(p.id, owner);
                }

                visit::walk_pat(self, p)
            }

            // HACK(or_patterns; Centril | dlrobertson): Avoid creating
            // HIR  nodes for `PatKind::Or` for the top level of a `ast::Arm`.
            // This is a temporary hack that should go away once we push down
            // `arm.pats: HirVec<P<Pat>>` -> `arm.pat: P<Pat>` to HIR. // Centril
            fn visit_arm(&mut self, arm: &'tcx Arm) {
                match &arm.pat.node {
                    PatKind::Or(pats) => pats.iter().for_each(|p| self.visit_pat(p)),
                    _ => self.visit_pat(&arm.pat),
                }
                walk_list!(self, visit_expr, &arm.guard);
                self.visit_expr(&arm.body);
                walk_list!(self, visit_attribute, &arm.attrs);
            }

            // HACK(or_patterns; Centril | dlrobertson): Same as above. // Centril
            fn visit_expr(&mut self, e: &'tcx Expr) {
                if let ExprKind::Let(pat, scrutinee) = &e.node {
                    walk_list!(self, visit_attribute, e.attrs.iter());
                    match &pat.node {
                        PatKind::Or(pats) => pats.iter().for_each(|p| self.visit_pat(p)),
                        _ => self.visit_pat(&pat),
                    }
                    self.visit_expr(scrutinee);
                    self.visit_expr_post(e);
                    return;
                }
                visit::walk_expr(self, e)
            }

            fn visit_item(&mut self, item: &'tcx Item) {
                let hir_id = self.lctx.allocate_hir_id_counter(item.id);

                match item.node {
                    ItemKind::Struct(_, ref generics)
                    | ItemKind::Union(_, ref generics)
                    | ItemKind::Enum(_, ref generics)
                    | ItemKind::TyAlias(_, ref generics)
                    | ItemKind::OpaqueTy(_, ref generics)
                    | ItemKind::Trait(_, _, ref generics, ..) => {
                        let def_id = self.lctx.resolver.definitions().local_def_id(item.id);
                        let count = generics
                            .params
                            .iter()
                            .filter(|param| match param.kind {
                                ast::GenericParamKind::Lifetime { .. } => true,
                                _ => false,
                            })
                            .count();
                        self.lctx.type_def_lifetime_params.insert(def_id, count);
                    }
                    ItemKind::Use(ref use_tree) => {
                        self.allocate_use_tree_hir_id_counters(use_tree, hir_id.owner);
                    }
                    _ => {}
                }

                self.with_hir_id_owner(Some(item.id), |this| {
                    visit::walk_item(this, item);
                });
            }

            fn visit_trait_item(&mut self, item: &'tcx TraitItem) {
                self.lctx.allocate_hir_id_counter(item.id);

                match item.node {
                    TraitItemKind::Method(_, None) => {
                        // Ignore patterns in trait methods without bodies
                        self.with_hir_id_owner(None, |this| {
                            visit::walk_trait_item(this, item)
                        });
                    }
                    _ => self.with_hir_id_owner(Some(item.id), |this| {
                        visit::walk_trait_item(this, item);
                    })
                }
            }

            fn visit_impl_item(&mut self, item: &'tcx ImplItem) {
                self.lctx.allocate_hir_id_counter(item.id);
                self.with_hir_id_owner(Some(item.id), |this| {
                    visit::walk_impl_item(this, item);
                });
            }

            fn visit_foreign_item(&mut self, i: &'tcx ForeignItem) {
                // Ignore patterns in foreign items
                self.with_hir_id_owner(None, |this| {
                    visit::walk_foreign_item(this, i)
                });
            }

            fn visit_ty(&mut self, t: &'tcx Ty) {
                match t.node {
                    // Mirrors the case in visit::walk_ty
                    TyKind::BareFn(ref f) => {
                        walk_list!(
                            self,
                            visit_generic_param,
                            &f.generic_params
                        );
                        // Mirrors visit::walk_fn_decl
                        for parameter in &f.decl.inputs {
                            // We don't lower the ids of argument patterns
                            self.with_hir_id_owner(None, |this| {
                                this.visit_pat(&parameter.pat);
                            });
                            self.visit_ty(&parameter.ty)
                        }
                        self.visit_fn_ret_ty(&f.decl.output)
                    }
                    _ => visit::walk_ty(self, t),
                }
            }
        }

        self.lower_node_id(CRATE_NODE_ID);
        debug_assert!(self.node_id_to_hir_id[CRATE_NODE_ID] == hir::CRATE_HIR_ID);

        visit::walk_crate(&mut MiscCollector { lctx: &mut self, hir_id_owner: None }, c);
        visit::walk_crate(&mut item::ItemLowerer { lctx: &mut self }, c);

        let module = self.lower_mod(&c.module);
        let attrs = self.lower_attrs(&c.attrs);
        let body_ids = body_ids(&self.bodies);

        self.resolver
            .definitions()
            .init_node_id_to_hir_id_mapping(self.node_id_to_hir_id);

        hir::Crate {
            module,
            attrs,
            span: c.span,
            exported_macros: hir::HirVec::from(self.exported_macros),
            non_exported_macro_attrs: hir::HirVec::from(self.non_exported_macro_attrs),
            items: self.items,
            trait_items: self.trait_items,
            impl_items: self.impl_items,
            bodies: self.bodies,
            body_ids,
            trait_impls: self.trait_impls,
            modules: self.modules,
        }
    }

    fn insert_item(&mut self, item: hir::Item) {
        let id = item.hir_id;
        // FIXME: Use `debug_asset-rt`.
        assert_eq!(id.local_id, hir::ItemLocalId::from_u32(0));
        self.items.insert(id, item);
        self.modules.get_mut(&self.current_module).unwrap().items.insert(id);
    }

    fn allocate_hir_id_counter(&mut self, owner: NodeId) -> hir::HirId {
        // Set up the counter if needed.
        self.item_local_id_counters.entry(owner).or_insert(0);
        // Always allocate the first `HirId` for the owner itself.
        let lowered = self.lower_node_id_with_owner(owner, owner);
        debug_assert_eq!(lowered.local_id.as_u32(), 0);
        lowered
    }

    fn lower_node_id_generic<F>(&mut self, ast_node_id: NodeId, alloc_hir_id: F) -> hir::HirId
    where
        F: FnOnce(&mut Self) -> hir::HirId,
    {
        if ast_node_id == DUMMY_NODE_ID {
            return hir::DUMMY_HIR_ID;
        }

        let min_size = ast_node_id.as_usize() + 1;

        if min_size > self.node_id_to_hir_id.len() {
            self.node_id_to_hir_id.resize(min_size, hir::DUMMY_HIR_ID);
        }

        let existing_hir_id = self.node_id_to_hir_id[ast_node_id];

        if existing_hir_id == hir::DUMMY_HIR_ID {
            // Generate a new `HirId`.
            let hir_id = alloc_hir_id(self);
            self.node_id_to_hir_id[ast_node_id] = hir_id;

            hir_id
        } else {
            existing_hir_id
        }
    }

    fn with_hir_id_owner<F, T>(&mut self, owner: NodeId, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let counter = self.item_local_id_counters
            .insert(owner, HIR_ID_COUNTER_LOCKED)
            .unwrap_or_else(|| panic!("no `item_local_id_counters` entry for {:?}", owner));
        let def_index = self.resolver.definitions().opt_def_index(owner).unwrap();
        self.current_hir_id_owner.push((def_index, counter));
        let ret = f(self);
        let (new_def_index, new_counter) = self.current_hir_id_owner.pop().unwrap();

        debug_assert!(def_index == new_def_index);
        debug_assert!(new_counter >= counter);

        let prev = self.item_local_id_counters
            .insert(owner, new_counter)
            .unwrap();
        debug_assert!(prev == HIR_ID_COUNTER_LOCKED);
        ret
    }

    /// This method allocates a new `HirId` for the given `NodeId` and stores it in
    /// the `LoweringContext`'s `NodeId => HirId` map.
    /// Take care not to call this method if the resulting `HirId` is then not
    /// actually used in the HIR, as that would trigger an assertion in the
    /// `HirIdValidator` later on, which makes sure that all `NodeId`s got mapped
    /// properly. Calling the method twice with the same `NodeId` is fine though.
    fn lower_node_id(&mut self, ast_node_id: NodeId) -> hir::HirId {
        self.lower_node_id_generic(ast_node_id, |this| {
            let &mut (def_index, ref mut local_id_counter) =
                this.current_hir_id_owner.last_mut().unwrap();
            let local_id = *local_id_counter;
            *local_id_counter += 1;
            hir::HirId {
                owner: def_index,
                local_id: hir::ItemLocalId::from_u32(local_id),
            }
        })
    }

    fn lower_node_id_with_owner(&mut self, ast_node_id: NodeId, owner: NodeId) -> hir::HirId {
        self.lower_node_id_generic(ast_node_id, |this| {
            let local_id_counter = this
                .item_local_id_counters
                .get_mut(&owner)
                .expect("called `lower_node_id_with_owner` before `allocate_hir_id_counter`");
            let local_id = *local_id_counter;

            // We want to be sure not to modify the counter in the map while it
            // is also on the stack. Otherwise we'll get lost updates when writing
            // back from the stack to the map.
            debug_assert!(local_id != HIR_ID_COUNTER_LOCKED);

            *local_id_counter += 1;
            let def_index = this
                .resolver
                .definitions()
                .opt_def_index(owner)
                .expect("you forgot to call `create_def_with_parent` or are lowering node-IDs \
                         that do not belong to the current owner");

            hir::HirId {
                owner: def_index,
                local_id: hir::ItemLocalId::from_u32(local_id),
            }
        })
    }

    fn next_id(&mut self) -> hir::HirId {
        self.lower_node_id(self.sess.next_node_id())
    }

    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        res.map_id(|id| {
            self.lower_node_id_generic(id, |_| {
                panic!("expected `NodeId` to be lowered already for res {:#?}", res);
            })
        })
    }

    fn expect_full_res(&mut self, id: NodeId) -> Res<NodeId> {
        self.resolver.get_partial_res(id).map_or(Res::Err, |pr| {
            if pr.unresolved_segments() != 0 {
                bug!("path not fully resolved: {:?}", pr);
            }
            pr.base_res()
        })
    }

    fn expect_full_res_from_use(&mut self, id: NodeId) -> impl Iterator<Item = Res<NodeId>> {
        self.resolver.get_import_res(id).present_items()
    }

    fn diagnostic(&self) -> &errors::Handler {
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
        span.fresh_expansion(ExpnData {
            allow_internal_unstable,
            ..ExpnData::default(ExpnKind::Desugaring(reason), span, self.sess.edition())
        })
    }

    fn with_anonymous_lifetime_mode<R>(
        &mut self,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        op: impl FnOnce(&mut Self) -> R,
    ) -> R {
        debug!(
            "with_anonymous_lifetime_mode(anonymous_lifetime_mode={:?})",
            anonymous_lifetime_mode,
        );
        let old_anonymous_lifetime_mode = self.anonymous_lifetime_mode;
        self.anonymous_lifetime_mode = anonymous_lifetime_mode;
        let result = op(self);
        self.anonymous_lifetime_mode = old_anonymous_lifetime_mode;
        debug!("with_anonymous_lifetime_mode: restoring anonymous_lifetime_mode={:?}",
               old_anonymous_lifetime_mode);
        result
    }

    /// Creates a new `hir::GenericParam` for every new lifetime and
    /// type parameter encountered while evaluating `f`. Definitions
    /// are created with the parent provided. If no `parent_id` is
    /// provided, no definitions will be returned.
    ///
    /// Presuming that in-band lifetimes are enabled, then
    /// `self.anonymous_lifetime_mode` will be updated to match the
    /// parameter while `f` is running (and restored afterwards).
    fn collect_in_band_defs<T, F>(
        &mut self,
        parent_id: DefId,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        f: F,
    ) -> (Vec<hir::GenericParam>, T)
    where
        F: FnOnce(&mut LoweringContext<'_>) -> (Vec<hir::GenericParam>, T),
    {
        assert!(!self.is_collecting_in_band_lifetimes);
        assert!(self.lifetimes_to_define.is_empty());
        let old_anonymous_lifetime_mode = self.anonymous_lifetime_mode;

        self.anonymous_lifetime_mode = anonymous_lifetime_mode;
        self.is_collecting_in_band_lifetimes = true;

        let (in_band_ty_params, res) = f(self);

        self.is_collecting_in_band_lifetimes = false;
        self.anonymous_lifetime_mode = old_anonymous_lifetime_mode;

        let lifetimes_to_define = self.lifetimes_to_define.split_off(0);

        let params = lifetimes_to_define
            .into_iter()
            .map(|(span, hir_name)| self.lifetime_to_generic_param(
                span, hir_name, parent_id.index,
            ))
            .chain(in_band_ty_params.into_iter())
            .collect();

        (params, res)
    }

    /// Converts a lifetime into a new generic parameter.
    fn lifetime_to_generic_param(
        &mut self,
        span: Span,
        hir_name: ParamName,
        parent_index: DefIndex,
    ) -> hir::GenericParam {
        let node_id = self.sess.next_node_id();

        // Get the name we'll use to make the def-path. Note
        // that collisions are ok here and this shouldn't
        // really show up for end-user.
        let (str_name, kind) = match hir_name {
            ParamName::Plain(ident) => (
                ident.as_interned_str(),
                hir::LifetimeParamKind::InBand,
            ),
            ParamName::Fresh(_) => (
                kw::UnderscoreLifetime.as_interned_str(),
                hir::LifetimeParamKind::Elided,
            ),
            ParamName::Error => (
                kw::UnderscoreLifetime.as_interned_str(),
                hir::LifetimeParamKind::Error,
            ),
        };

        // Add a definition for the in-band lifetime def.
        self.resolver.definitions().create_def_with_parent(
            parent_index,
            node_id,
            DefPathData::LifetimeNs(str_name),
            ExpnId::root(),
            span,
        );

        hir::GenericParam {
            hir_id: self.lower_node_id(node_id),
            name: hir_name,
            attrs: hir_vec![],
            bounds: hir_vec![],
            span,
            pure_wrt_drop: false,
            kind: hir::GenericParamKind::Lifetime { kind }
        }
    }

    /// When there is a reference to some lifetime `'a`, and in-band
    /// lifetimes are enabled, then we want to push that lifetime into
    /// the vector of names to define later. In that case, it will get
    /// added to the appropriate generics.
    fn maybe_collect_in_band_lifetime(&mut self, ident: Ident) {
        if !self.is_collecting_in_band_lifetimes {
            return;
        }

        if !self.sess.features_untracked().in_band_lifetimes {
            return;
        }

        if self.in_scope_lifetimes.contains(&ParamName::Plain(ident.modern())) {
            return;
        }

        let hir_name = ParamName::Plain(ident);

        if self.lifetimes_to_define.iter()
                                   .any(|(_, lt_name)| lt_name.modern() == hir_name.modern()) {
            return;
        }

        self.lifetimes_to_define.push((ident.span, hir_name));
    }

    /// When we have either an elided or `'_` lifetime in an impl
    /// header, we convert it to an in-band lifetime.
    fn collect_fresh_in_band_lifetime(&mut self, span: Span) -> ParamName {
        assert!(self.is_collecting_in_band_lifetimes);
        let index = self.lifetimes_to_define.len();
        let hir_name = ParamName::Fresh(index);
        self.lifetimes_to_define.push((span, hir_name));
        hir_name
    }

    // Evaluates `f` with the lifetimes in `params` in-scope.
    // This is used to track which lifetimes have already been defined, and
    // which are new in-band lifetimes that need to have a definition created
    // for them.
    fn with_in_scope_lifetime_defs<T, F>(&mut self, params: &[GenericParam], f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let old_len = self.in_scope_lifetimes.len();
        let lt_def_names = params.iter().filter_map(|param| match param.kind {
            GenericParamKind::Lifetime { .. } => Some(ParamName::Plain(param.ident.modern())),
            _ => None,
        });
        self.in_scope_lifetimes.extend(lt_def_names);

        let res = f(self);

        self.in_scope_lifetimes.truncate(old_len);
        res
    }

    /// Appends in-band lifetime defs and argument-position `impl
    /// Trait` defs to the existing set of generics.
    ///
    /// Presuming that in-band lifetimes are enabled, then
    /// `self.anonymous_lifetime_mode` will be updated to match the
    /// parameter while `f` is running (and restored afterwards).
    fn add_in_band_defs<F, T>(
        &mut self,
        generics: &Generics,
        parent_id: DefId,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        f: F,
    ) -> (hir::Generics, T)
    where
        F: FnOnce(&mut LoweringContext<'_>, &mut Vec<hir::GenericParam>) -> T,
    {
        let (in_band_defs, (mut lowered_generics, res)) = self.with_in_scope_lifetime_defs(
            &generics.params,
            |this| {
                this.collect_in_band_defs(parent_id, anonymous_lifetime_mode, |this| {
                    let mut params = Vec::new();
                    // Note: it is necessary to lower generics *before* calling `f`.
                    // When lowering `async fn`, there's a final step when lowering
                    // the return type that assumes that all in-scope lifetimes have
                    // already been added to either `in_scope_lifetimes` or
                    // `lifetimes_to_define`. If we swapped the order of these two,
                    // in-band-lifetimes introduced by generics or where-clauses
                    // wouldn't have been added yet.
                    let generics = this.lower_generics(
                        generics,
                        ImplTraitContext::Universal(&mut params),
                    );
                    let res = f(this, &mut params);
                    (params, (generics, res))
                })
            },
        );

        let mut lowered_params: Vec<_> = lowered_generics
            .params
            .into_iter()
            .chain(in_band_defs)
            .collect();

        // FIXME(const_generics): the compiler doesn't always cope with
        // unsorted generic parameters at the moment, so we make sure
        // that they're ordered correctly here for now. (When we chain
        // the `in_band_defs`, we might make the order unsorted.)
        lowered_params.sort_by_key(|param| {
            match param.kind {
                hir::GenericParamKind::Lifetime { .. } => ParamKindOrd::Lifetime,
                hir::GenericParamKind::Type { .. } => ParamKindOrd::Type,
                hir::GenericParamKind::Const { .. } => ParamKindOrd::Const,
            }
        });

        lowered_generics.params = lowered_params.into();

        (lowered_generics, res)
    }

    fn with_dyn_type_scope<T, F>(&mut self, in_scope: bool, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let was_in_dyn_type = self.is_in_dyn_type;
        self.is_in_dyn_type = in_scope;

        let result = f(self);

        self.is_in_dyn_type = was_in_dyn_type;

        result
    }

    fn with_new_scopes<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let catch_scopes = mem::take(&mut self.catch_scopes);
        let loop_scopes = mem::take(&mut self.loop_scopes);
        let ret = f(self);
        self.catch_scopes = catch_scopes;
        self.loop_scopes = loop_scopes;

        self.is_in_loop_condition = was_in_loop_condition;

        ret
    }

    fn def_key(&mut self, id: DefId) -> DefKey {
        if id.is_local() {
            self.resolver.definitions().def_key(id.index)
        } else {
            self.cstore.def_key(id)
        }
    }

    fn lower_attrs_extendable(&mut self, attrs: &[Attribute]) -> Vec<Attribute> {
        attrs
            .iter()
            .map(|a| self.lower_attr(a))
            .collect()
    }

    fn lower_attrs(&mut self, attrs: &[Attribute]) -> hir::HirVec<Attribute> {
        self.lower_attrs_extendable(attrs).into()
    }

    fn lower_attr(&mut self, attr: &Attribute) -> Attribute {
        // Note that we explicitly do not walk the path. Since we don't really
        // lower attributes (we use the AST version) there is nowhere to keep
        // the `HirId`s. We don't actually need HIR version of attributes anyway.
        Attribute {
            id: attr.id,
            style: attr.style,
            path: attr.path.clone(),
            tokens: self.lower_token_stream(attr.tokens.clone()),
            is_sugared_doc: attr.is_sugared_doc,
            span: attr.span,
        }
    }

    fn lower_token_stream(&mut self, tokens: TokenStream) -> TokenStream {
        tokens
            .into_trees()
            .flat_map(|tree| self.lower_token_tree(tree).into_trees())
            .collect()
    }

    fn lower_token_tree(&mut self, tree: TokenTree) -> TokenStream {
        match tree {
            TokenTree::Token(token) => self.lower_token(token),
            TokenTree::Delimited(span, delim, tts) => TokenTree::Delimited(
                span,
                delim,
                self.lower_token_stream(tts),
            ).into(),
        }
    }

    fn lower_token(&mut self, token: Token) -> TokenStream {
        match token.kind {
            token::Interpolated(nt) => {
                let tts = nt.to_tokenstream(&self.sess.parse_sess, token.span);
                self.lower_token_stream(tts)
            }
            _ => TokenTree::Token(token).into(),
        }
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
        constraint: &AssocTyConstraint,
        itctx: ImplTraitContext<'_>,
    ) -> hir::TypeBinding {
        debug!("lower_assoc_ty_constraint(constraint={:?}, itctx={:?})", constraint, itctx);

        let kind = match constraint.kind {
            AssocTyConstraintKind::Equality { ref ty } => hir::TypeBindingKind::Equality {
                ty: self.lower_ty(ty, itctx)
            },
            AssocTyConstraintKind::Bound { ref bounds } => {
                // Piggy-back on the `impl Trait` context to figure out the correct behavior.
                let (desugar_to_impl_trait, itctx) = match itctx {
                    // We are in the return position:
                    //
                    //     fn foo() -> impl Iterator<Item: Debug>
                    //
                    // so desugar to
                    //
                    //     fn foo() -> impl Iterator<Item = impl Debug>
                    ImplTraitContext::OpaqueTy(_) => (true, itctx),

                    // We are in the argument position, but within a dyn type:
                    //
                    //     fn foo(x: dyn Iterator<Item: Debug>)
                    //
                    // so desugar to
                    //
                    //     fn foo(x: dyn Iterator<Item = impl Debug>)
                    ImplTraitContext::Universal(_) if self.is_in_dyn_type => (true, itctx),

                    // In `type Foo = dyn Iterator<Item: Debug>` we desugar to
                    // `type Foo = dyn Iterator<Item = impl Debug>` but we have to override the
                    // "impl trait context" to permit `impl Debug` in this position (it desugars
                    // then to an opaque type).
                    //
                    // FIXME: this is only needed until `impl Trait` is allowed in type aliases.
                    ImplTraitContext::Disallowed(_) if self.is_in_dyn_type =>
                        (true, ImplTraitContext::OpaqueTy(None)),

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

                    let impl_trait_node_id = self.sess.next_node_id();
                    let parent_def_index = self.current_hir_id_owner.last().unwrap().0;
                    self.resolver.definitions().create_def_with_parent(
                        parent_def_index,
                        impl_trait_node_id,
                        DefPathData::ImplTrait,
                        ExpnId::root(),
                        constraint.span,
                    );

                    self.with_dyn_type_scope(false, |this| {
                        let ty = this.lower_ty(
                            &Ty {
                                id: this.sess.next_node_id(),
                                node: TyKind::ImplTrait(impl_trait_node_id, bounds.clone()),
                                span: constraint.span,
                            },
                            itctx,
                        );

                        hir::TypeBindingKind::Equality {
                            ty
                        }
                    })
                } else {
                    // Desugar `AssocTy: Bounds` into a type binding where the
                    // later desugars into a trait predicate.
                    let bounds = self.lower_param_bounds(bounds, itctx);

                    hir::TypeBindingKind::Constraint {
                        bounds
                    }
                }
            }
        };

        hir::TypeBinding {
            hir_id: self.lower_node_id(constraint.id),
            ident: constraint.ident,
            kind,
            span: constraint.span,
        }
    }

    fn lower_generic_arg(&mut self,
                         arg: &ast::GenericArg,
                         itctx: ImplTraitContext<'_>)
                         -> hir::GenericArg {
        match arg {
            ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.lower_lifetime(&lt)),
            ast::GenericArg::Type(ty) => GenericArg::Type(self.lower_ty_direct(&ty, itctx)),
            ast::GenericArg::Const(ct) => {
                GenericArg::Const(ConstArg {
                    value: self.lower_anon_const(&ct),
                    span: ct.value.span,
                })
            }
        }
    }

    fn lower_ty(&mut self, t: &Ty, itctx: ImplTraitContext<'_>) -> P<hir::Ty> {
        P(self.lower_ty_direct(t, itctx))
    }

    fn lower_path_ty(
        &mut self,
        t: &Ty,
        qself: &Option<QSelf>,
        path: &Path,
        param_mode: ParamMode,
        itctx: ImplTraitContext<'_>
    ) -> hir::Ty {
        let id = self.lower_node_id(t.id);
        let qpath = self.lower_qpath(t.id, qself, path, param_mode, itctx);
        let ty = self.ty_path(id, t.span, qpath);
        if let hir::TyKind::TraitObject(..) = ty.node {
            self.maybe_lint_bare_trait(t.span, t.id, qself.is_none() && path.is_global());
        }
        ty
    }

    fn lower_ty_direct(&mut self, t: &Ty, mut itctx: ImplTraitContext<'_>) -> hir::Ty {
        let kind = match t.node {
            TyKind::Infer => hir::TyKind::Infer,
            TyKind::Err => hir::TyKind::Err,
            TyKind::Slice(ref ty) => hir::TyKind::Slice(self.lower_ty(ty, itctx)),
            TyKind::Ptr(ref mt) => hir::TyKind::Ptr(self.lower_mt(mt, itctx)),
            TyKind::Rptr(ref region, ref mt) => {
                let span = self.sess.source_map().next_point(t.span.shrink_to_lo());
                let lifetime = match *region {
                    Some(ref lt) => self.lower_lifetime(lt),
                    None => self.elided_ref_lifetime(span),
                };
                hir::TyKind::Rptr(lifetime, self.lower_mt(mt, itctx))
            }
            TyKind::BareFn(ref f) => self.with_in_scope_lifetime_defs(
                &f.generic_params,
                |this| {
                    this.with_anonymous_lifetime_mode(
                        AnonymousLifetimeMode::PassThrough,
                        |this| {
                            hir::TyKind::BareFn(P(hir::BareFnTy {
                                generic_params: this.lower_generic_params(
                                    &f.generic_params,
                                    &NodeMap::default(),
                                    ImplTraitContext::disallowed(),
                                ),
                                unsafety: this.lower_unsafety(f.unsafety),
                                abi: f.abi,
                                decl: this.lower_fn_decl(&f.decl, None, false, None),
                                param_names: this.lower_fn_params_to_names(&f.decl),
                            }))
                        },
                    )
                },
            ),
            TyKind::Never => hir::TyKind::Never,
            TyKind::Tup(ref tys) => {
                hir::TyKind::Tup(tys.iter().map(|ty| {
                    self.lower_ty_direct(ty, itctx.reborrow())
                }).collect())
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
                    P(hir::Path {
                        res,
                        segments: hir_vec![hir::PathSegment::from_ident(
                            Ident::with_dummy_span(kw::SelfUpper)
                        )],
                        span: t.span,
                    }),
                ))
            },
            TyKind::Array(ref ty, ref length) => {
                hir::TyKind::Array(self.lower_ty(ty, itctx), self.lower_anon_const(length))
            }
            TyKind::Typeof(ref expr) => {
                hir::TyKind::Typeof(self.lower_anon_const(expr))
            }
            TyKind::TraitObject(ref bounds, kind) => {
                let mut lifetime_bound = None;
                let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                    let bounds = bounds
                        .iter()
                        .filter_map(|bound| match *bound {
                            GenericBound::Trait(ref ty, TraitBoundModifier::None) => {
                                Some(this.lower_poly_trait_ref(ty, itctx.reborrow()))
                            }
                            GenericBound::Trait(_, TraitBoundModifier::Maybe) => None,
                            GenericBound::Outlives(ref lifetime) => {
                                if lifetime_bound.is_none() {
                                    lifetime_bound = Some(this.lower_lifetime(lifetime));
                                }
                                None
                            }
                        })
                        .collect();
                    let lifetime_bound =
                        lifetime_bound.unwrap_or_else(|| this.elided_dyn_bound(t.span));
                    (bounds, lifetime_bound)
                });
                if kind != TraitObjectSyntax::Dyn {
                    self.maybe_lint_bare_trait(t.span, t.id, false);
                }
                hir::TyKind::TraitObject(bounds, lifetime_bound)
            }
            TyKind::ImplTrait(def_node_id, ref bounds) => {
                let span = t.span;
                match itctx {
                    ImplTraitContext::OpaqueTy(fn_def_id) => {
                        self.lower_opaque_impl_trait(
                            span, fn_def_id, def_node_id,
                            |this| this.lower_param_bounds(bounds, itctx),
                        )
                    }
                    ImplTraitContext::Universal(in_band_ty_params) => {
                        // Add a definition for the in-band `Param`.
                        let def_index = self
                            .resolver
                            .definitions()
                            .opt_def_index(def_node_id)
                            .unwrap();

                        let hir_bounds = self.lower_param_bounds(
                            bounds,
                            ImplTraitContext::Universal(in_band_ty_params),
                        );
                        // Set the name to `impl Bound1 + Bound2`.
                        let ident = Ident::from_str_and_span(&pprust::ty_to_string(t), span);
                        in_band_ty_params.push(hir::GenericParam {
                            hir_id: self.lower_node_id(def_node_id),
                            name: ParamName::Plain(ident),
                            pure_wrt_drop: false,
                            attrs: hir_vec![],
                            bounds: hir_bounds,
                            span,
                            kind: hir::GenericParamKind::Type {
                                default: None,
                                synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
                            }
                        });

                        hir::TyKind::Path(hir::QPath::Resolved(
                            None,
                            P(hir::Path {
                                span,
                                res: Res::Def(DefKind::TyParam, DefId::local(def_index)),
                                segments: hir_vec![hir::PathSegment::from_ident(ident)],
                            }),
                        ))
                    }
                    ImplTraitContext::Disallowed(pos) => {
                        let allowed_in = if self.sess.features_untracked()
                                                .impl_trait_in_bindings {
                            "bindings or function and inherent method return types"
                        } else {
                            "function and inherent method return types"
                        };
                        let mut err = struct_span_err!(
                            self.sess,
                            t.span,
                            E0562,
                            "`impl Trait` not allowed outside of {}",
                            allowed_in,
                        );
                        if pos == ImplTraitPosition::Binding &&
                            nightly_options::is_nightly_build() {
                            help!(err,
                                  "add `#![feature(impl_trait_in_bindings)]` to the crate \
                                   attributes to enable");
                        }
                        err.emit();
                        hir::TyKind::Err
                    }
                }
            }
            TyKind::Mac(_) => bug!("`TyMac` should have been expanded by now"),
            TyKind::CVarArgs => {
                // Create the implicit lifetime of the "spoofed" `VaListImpl`.
                let span = self.sess.source_map().next_point(t.span.shrink_to_lo());
                let lt = self.new_implicit_lifetime(span);
                hir::TyKind::CVarArgs(lt)
            },
        };

        hir::Ty {
            node: kind,
            span: t.span,
            hir_id: self.lower_node_id(t.id),
        }
    }

    fn lower_opaque_impl_trait(
        &mut self,
        span: Span,
        fn_def_id: Option<DefId>,
        opaque_ty_node_id: NodeId,
        lower_bounds: impl FnOnce(&mut LoweringContext<'_>) -> hir::GenericBounds,
    ) -> hir::TyKind {
        debug!(
            "lower_opaque_impl_trait(fn_def_id={:?}, opaque_ty_node_id={:?}, span={:?})",
            fn_def_id,
            opaque_ty_node_id,
            span,
        );

        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let opaque_ty_span = self.mark_span_with_reason(
            DesugaringKind::OpaqueTy,
            span,
            None,
        );

        let opaque_ty_def_index = self
            .resolver
            .definitions()
            .opt_def_index(opaque_ty_node_id)
            .unwrap();

        self.allocate_hir_id_counter(opaque_ty_node_id);

        let hir_bounds = self.with_hir_id_owner(opaque_ty_node_id, lower_bounds);

        let (lifetimes, lifetime_defs) = self.lifetimes_from_impl_trait_bounds(
            opaque_ty_node_id,
            opaque_ty_def_index,
            &hir_bounds,
        );

        debug!(
            "lower_opaque_impl_trait: lifetimes={:#?}", lifetimes,
        );

        debug!(
            "lower_opaque_impl_trait: lifetime_defs={:#?}", lifetime_defs,
        );

        self.with_hir_id_owner(opaque_ty_node_id, |lctx| {
            let opaque_ty_item = hir::OpaqueTy {
                generics: hir::Generics {
                    params: lifetime_defs,
                    where_clause: hir::WhereClause {
                        predicates: hir_vec![],
                        span,
                    },
                    span,
                },
                bounds: hir_bounds,
                impl_trait_fn: fn_def_id,
                origin: hir::OpaqueTyOrigin::FnReturn,
            };

            trace!("lower_opaque_impl_trait: {:#?}", opaque_ty_def_index);
            let opaque_ty_id = lctx.generate_opaque_type(
                opaque_ty_node_id,
                opaque_ty_item,
                span,
                opaque_ty_span,
            );

            // `impl Trait` now just becomes `Foo<'a, 'b, ..>`.
            hir::TyKind::Def(hir::ItemId { id: opaque_ty_id }, lifetimes)
        })
    }

    /// Registers a new opaque type with the proper `NodeId`s and
    /// returns the lowered node-ID for the opaque type.
    fn generate_opaque_type(
        &mut self,
        opaque_ty_node_id: NodeId,
        opaque_ty_item: hir::OpaqueTy,
        span: Span,
        opaque_ty_span: Span,
    ) -> hir::HirId {
        let opaque_ty_item_kind = hir::ItemKind::OpaqueTy(opaque_ty_item);
        let opaque_ty_id = self.lower_node_id(opaque_ty_node_id);
        // Generate an `type Foo = impl Trait;` declaration.
        trace!("registering opaque type with id {:#?}", opaque_ty_id);
        let opaque_ty_item = hir::Item {
            hir_id: opaque_ty_id,
            ident: Ident::invalid(),
            attrs: Default::default(),
            node: opaque_ty_item_kind,
            vis: respan(span.shrink_to_lo(), hir::VisibilityKind::Inherited),
            span: opaque_ty_span,
        };

        // Insert the item into the global item list. This usually happens
        // automatically for all AST items. But this opaque type item
        // does not actually exist in the AST.
        self.insert_item(opaque_ty_item);
        opaque_ty_id
    }

    fn lifetimes_from_impl_trait_bounds(
        &mut self,
        opaque_ty_id: NodeId,
        parent_index: DefIndex,
        bounds: &hir::GenericBounds,
    ) -> (HirVec<hir::GenericArg>, HirVec<hir::GenericParam>) {
        debug!(
            "lifetimes_from_impl_trait_bounds(opaque_ty_id={:?}, \
             parent_index={:?}, \
             bounds={:#?})",
            opaque_ty_id, parent_index, bounds,
        );

        // This visitor walks over `impl Trait` bounds and creates defs for all lifetimes that
        // appear in the bounds, excluding lifetimes that are created within the bounds.
        // E.g., `'a`, `'b`, but not `'c` in `impl for<'c> SomeTrait<'a, 'b, 'c>`.
        struct ImplTraitLifetimeCollector<'r, 'a> {
            context: &'r mut LoweringContext<'a>,
            parent: DefIndex,
            opaque_ty_id: NodeId,
            collect_elided_lifetimes: bool,
            currently_bound_lifetimes: Vec<hir::LifetimeName>,
            already_defined_lifetimes: FxHashSet<hir::LifetimeName>,
            output_lifetimes: Vec<hir::GenericArg>,
            output_lifetime_params: Vec<hir::GenericParam>,
        }

        impl<'r, 'a, 'v> hir::intravisit::Visitor<'v> for ImplTraitLifetimeCollector<'r, 'a> {
            fn nested_visit_map<'this>(
                &'this mut self,
            ) -> hir::intravisit::NestedVisitorMap<'this, 'v> {
                hir::intravisit::NestedVisitorMap::None
            }

            fn visit_generic_args(&mut self, span: Span, parameters: &'v hir::GenericArgs) {
                // Don't collect elided lifetimes used inside of `Fn()` syntax.
                if parameters.parenthesized {
                    let old_collect_elided_lifetimes = self.collect_elided_lifetimes;
                    self.collect_elided_lifetimes = false;
                    hir::intravisit::walk_generic_args(self, span, parameters);
                    self.collect_elided_lifetimes = old_collect_elided_lifetimes;
                } else {
                    hir::intravisit::walk_generic_args(self, span, parameters);
                }
            }

            fn visit_ty(&mut self, t: &'v hir::Ty) {
                // Don't collect elided lifetimes used inside of `fn()` syntax.
                if let hir::TyKind::BareFn(_) = t.node {
                    let old_collect_elided_lifetimes = self.collect_elided_lifetimes;
                    self.collect_elided_lifetimes = false;

                    // Record the "stack height" of `for<'a>` lifetime bindings
                    // to be able to later fully undo their introduction.
                    let old_len = self.currently_bound_lifetimes.len();
                    hir::intravisit::walk_ty(self, t);
                    self.currently_bound_lifetimes.truncate(old_len);

                    self.collect_elided_lifetimes = old_collect_elided_lifetimes;
                } else {
                    hir::intravisit::walk_ty(self, t)
                }
            }

            fn visit_poly_trait_ref(
                &mut self,
                trait_ref: &'v hir::PolyTraitRef,
                modifier: hir::TraitBoundModifier,
            ) {
                // Record the "stack height" of `for<'a>` lifetime bindings
                // to be able to later fully undo their introduction.
                let old_len = self.currently_bound_lifetimes.len();
                hir::intravisit::walk_poly_trait_ref(self, trait_ref, modifier);
                self.currently_bound_lifetimes.truncate(old_len);
            }

            fn visit_generic_param(&mut self, param: &'v hir::GenericParam) {
                // Record the introduction of 'a in `for<'a> ...`.
                if let hir::GenericParamKind::Lifetime { .. } = param.kind {
                    // Introduce lifetimes one at a time so that we can handle
                    // cases like `fn foo<'d>() -> impl for<'a, 'b: 'a, 'c: 'b + 'd>`.
                    let lt_name = hir::LifetimeName::Param(param.name);
                    self.currently_bound_lifetimes.push(lt_name);
                }

                hir::intravisit::walk_generic_param(self, param);
            }

            fn visit_lifetime(&mut self, lifetime: &'v hir::Lifetime) {
                let name = match lifetime.name {
                    hir::LifetimeName::Implicit | hir::LifetimeName::Underscore => {
                        if self.collect_elided_lifetimes {
                            // Use `'_` for both implicit and underscore lifetimes in
                            // `type Foo<'_> = impl SomeTrait<'_>;`.
                            hir::LifetimeName::Underscore
                        } else {
                            return;
                        }
                    }
                    hir::LifetimeName::Param(_) => lifetime.name,

                    // Refers to some other lifetime that is "in
                    // scope" within the type.
                    hir::LifetimeName::ImplicitObjectLifetimeDefault => return,

                    hir::LifetimeName::Error | hir::LifetimeName::Static => return,
                };

                if !self.currently_bound_lifetimes.contains(&name)
                    && !self.already_defined_lifetimes.contains(&name) {
                    self.already_defined_lifetimes.insert(name);

                    self.output_lifetimes.push(hir::GenericArg::Lifetime(hir::Lifetime {
                        hir_id: self.context.next_id(),
                        span: lifetime.span,
                        name,
                    }));

                    let def_node_id = self.context.sess.next_node_id();
                    let hir_id =
                        self.context.lower_node_id_with_owner(def_node_id, self.opaque_ty_id);
                    self.context.resolver.definitions().create_def_with_parent(
                        self.parent,
                        def_node_id,
                        DefPathData::LifetimeNs(name.ident().as_interned_str()),
                        ExpnId::root(),
                        lifetime.span);

                    let (name, kind) = match name {
                        hir::LifetimeName::Underscore => (
                            hir::ParamName::Plain(Ident::with_dummy_span(kw::UnderscoreLifetime)),
                            hir::LifetimeParamKind::Elided,
                        ),
                        hir::LifetimeName::Param(param_name) => (
                            param_name,
                            hir::LifetimeParamKind::Explicit,
                        ),
                        _ => bug!("expected `LifetimeName::Param` or `ParamName::Plain`"),
                    };

                    self.output_lifetime_params.push(hir::GenericParam {
                        hir_id,
                        name,
                        span: lifetime.span,
                        pure_wrt_drop: false,
                        attrs: hir_vec![],
                        bounds: hir_vec![],
                        kind: hir::GenericParamKind::Lifetime { kind }
                    });
                }
            }
        }

        let mut lifetime_collector = ImplTraitLifetimeCollector {
            context: self,
            parent: parent_index,
            opaque_ty_id,
            collect_elided_lifetimes: true,
            currently_bound_lifetimes: Vec::new(),
            already_defined_lifetimes: FxHashSet::default(),
            output_lifetimes: Vec::new(),
            output_lifetime_params: Vec::new(),
        };

        for bound in bounds {
            hir::intravisit::walk_param_bound(&mut lifetime_collector, &bound);
        }

        (
            lifetime_collector.output_lifetimes.into(),
            lifetime_collector.output_lifetime_params.into(),
        )
    }

    fn lower_qpath(
        &mut self,
        id: NodeId,
        qself: &Option<QSelf>,
        p: &Path,
        param_mode: ParamMode,
        mut itctx: ImplTraitContext<'_>,
    ) -> hir::QPath {
        let qself_position = qself.as_ref().map(|q| q.position);
        let qself = qself.as_ref().map(|q| self.lower_ty(&q.ty, itctx.reborrow()));

        let partial_res = self.resolver
            .get_partial_res(id)
            .unwrap_or_else(|| PartialRes::new(Res::Err));

        let proj_start = p.segments.len() - partial_res.unresolved_segments();
        let path = P(hir::Path {
            res: self.lower_res(partial_res.base_res()),
            segments: p.segments[..proj_start]
                .iter()
                .enumerate()
                .map(|(i, segment)| {
                    let param_mode = match (qself_position, param_mode) {
                        (Some(j), ParamMode::Optional) if i < j => {
                            // This segment is part of the trait path in a
                            // qualified path - one of `a`, `b` or `Trait`
                            // in `<X as a::b::Trait>::T::U::method`.
                            ParamMode::Explicit
                        }
                        _ => param_mode,
                    };

                    // Figure out if this is a type/trait segment,
                    // which may need lifetime elision performed.
                    let parent_def_id = |this: &mut Self, def_id: DefId| DefId {
                        krate: def_id.krate,
                        index: this.def_key(def_id).parent.expect("missing parent"),
                    };
                    let type_def_id = match partial_res.base_res() {
                        Res::Def(DefKind::AssocTy, def_id) if i + 2 == proj_start => {
                            Some(parent_def_id(self, def_id))
                        }
                        Res::Def(DefKind::Variant, def_id) if i + 1 == proj_start => {
                            Some(parent_def_id(self, def_id))
                        }
                        Res::Def(DefKind::Struct, def_id)
                        | Res::Def(DefKind::Union, def_id)
                        | Res::Def(DefKind::Enum, def_id)
                        | Res::Def(DefKind::TyAlias, def_id)
                        | Res::Def(DefKind::Trait, def_id) if i + 1 == proj_start =>
                        {
                            Some(def_id)
                        }
                        _ => None,
                    };
                    let parenthesized_generic_args = match partial_res.base_res() {
                        // `a::b::Trait(Args)`
                        Res::Def(DefKind::Trait, _)
                            if i + 1 == proj_start => ParenthesizedGenericArgs::Ok,
                        // `a::b::Trait(Args)::TraitItem`
                        Res::Def(DefKind::Method, _)
                        | Res::Def(DefKind::AssocConst, _)
                        | Res::Def(DefKind::AssocTy, _)
                            if i + 2 == proj_start =>
                        {
                            ParenthesizedGenericArgs::Ok
                        }
                        // Avoid duplicated errors.
                        Res::Err => ParenthesizedGenericArgs::Ok,
                        // An error
                        Res::Def(DefKind::Struct, _)
                        | Res::Def(DefKind::Enum, _)
                        | Res::Def(DefKind::Union, _)
                        | Res::Def(DefKind::TyAlias, _)
                        | Res::Def(DefKind::Variant, _) if i + 1 == proj_start =>
                        {
                            ParenthesizedGenericArgs::Err
                        }
                        // A warning for now, for compatibility reasons.
                        _ => ParenthesizedGenericArgs::Warn,
                    };

                    let num_lifetimes = type_def_id.map_or(0, |def_id| {
                        if let Some(&n) = self.type_def_lifetime_params.get(&def_id) {
                            return n;
                        }
                        assert!(!def_id.is_local());
                        let item_generics =
                            self.cstore.item_generics_cloned_untracked(def_id, self.sess);
                        let n = item_generics.own_counts().lifetimes;
                        self.type_def_lifetime_params.insert(def_id, n);
                        n
                    });
                    self.lower_path_segment(
                        p.span,
                        segment,
                        param_mode,
                        num_lifetimes,
                        parenthesized_generic_args,
                        itctx.reborrow(),
                        None,
                    )
                })
                .collect(),
            span: p.span,
        });

        // Simple case, either no projections, or only fully-qualified.
        // E.g., `std::mem::size_of` or `<I as Iterator>::Item`.
        if partial_res.unresolved_segments() == 0 {
            return hir::QPath::Resolved(qself, path);
        }

        // Create the innermost type that we're projecting from.
        let mut ty = if path.segments.is_empty() {
            // If the base path is empty that means there exists a
            // syntactical `Self`, e.g., `&i32` in `<&i32>::clone`.
            qself.expect("missing QSelf for <T>::...")
        } else {
            // Otherwise, the base path is an implicit `Self` type path,
            // e.g., `Vec` in `Vec::new` or `<I as Iterator>::Item` in
            // `<I as Iterator>::Item::default`.
            let new_id = self.next_id();
            P(self.ty_path(new_id, p.span, hir::QPath::Resolved(qself, path)))
        };

        // Anything after the base path are associated "extensions",
        // out of which all but the last one are associated types,
        // e.g., for `std::vec::Vec::<T>::IntoIter::Item::clone`:
        // * base path is `std::vec::Vec<T>`
        // * "extensions" are `IntoIter`, `Item` and `clone`
        // * type nodes are:
        //   1. `std::vec::Vec<T>` (created above)
        //   2. `<std::vec::Vec<T>>::IntoIter`
        //   3. `<<std::vec::Vec<T>>::IntoIter>::Item`
        // * final path is `<<<std::vec::Vec<T>>::IntoIter>::Item>::clone`
        for (i, segment) in p.segments.iter().enumerate().skip(proj_start) {
            let segment = P(self.lower_path_segment(
                p.span,
                segment,
                param_mode,
                0,
                ParenthesizedGenericArgs::Warn,
                itctx.reborrow(),
                None,
            ));
            let qpath = hir::QPath::TypeRelative(ty, segment);

            // It's finished, return the extension of the right node type.
            if i == p.segments.len() - 1 {
                return qpath;
            }

            // Wrap the associated extension in another type node.
            let new_id = self.next_id();
            ty = P(self.ty_path(new_id, p.span, qpath));
        }

        // We should've returned in the for loop above.
        span_bug!(
            p.span,
            "lower_qpath: no final extension segment in {}..{}",
            proj_start,
            p.segments.len()
        )
    }

    fn lower_path_extra(
        &mut self,
        res: Res,
        p: &Path,
        param_mode: ParamMode,
        explicit_owner: Option<NodeId>,
    ) -> hir::Path {
        hir::Path {
            res,
            segments: p.segments
                .iter()
                .map(|segment| {
                    self.lower_path_segment(
                        p.span,
                        segment,
                        param_mode,
                        0,
                        ParenthesizedGenericArgs::Err,
                        ImplTraitContext::disallowed(),
                        explicit_owner,
                    )
                })
                .collect(),
            span: p.span,
        }
    }

    fn lower_path(&mut self, id: NodeId, p: &Path, param_mode: ParamMode) -> hir::Path {
        let res = self.expect_full_res(id);
        let res = self.lower_res(res);
        self.lower_path_extra(res, p, param_mode, None)
    }

    fn lower_path_segment(
        &mut self,
        path_span: Span,
        segment: &PathSegment,
        param_mode: ParamMode,
        expected_lifetimes: usize,
        parenthesized_generic_args: ParenthesizedGenericArgs,
        itctx: ImplTraitContext<'_>,
        explicit_owner: Option<NodeId>,
    ) -> hir::PathSegment {
        let (mut generic_args, infer_args) = if let Some(ref generic_args) = segment.args {
            let msg = "parenthesized type parameters may only be used with a `Fn` trait";
            match **generic_args {
                GenericArgs::AngleBracketed(ref data) => {
                    self.lower_angle_bracketed_parameter_data(data, param_mode, itctx)
                }
                GenericArgs::Parenthesized(ref data) => match parenthesized_generic_args {
                    ParenthesizedGenericArgs::Ok => self.lower_parenthesized_parameter_data(data),
                    ParenthesizedGenericArgs::Warn => {
                        self.sess.buffer_lint(
                            PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES,
                            CRATE_NODE_ID,
                            data.span,
                            msg.into(),
                        );
                        (hir::GenericArgs::none(), true)
                    }
                    ParenthesizedGenericArgs::Err => {
                        let mut err = struct_span_err!(self.sess, data.span, E0214, "{}", msg);
                        err.span_label(data.span, "only `Fn` traits may use parentheses");
                        if let Ok(snippet) = self.sess.source_map().span_to_snippet(data.span) {
                            // Do not suggest going from `Trait()` to `Trait<>`
                            if data.inputs.len() > 0 {
                                let split = snippet.find('(').unwrap();
                                let trait_name = &snippet[0..split];
                                let args = &snippet[split + 1 .. snippet.len() - 1];
                                err.span_suggestion(
                                    data.span,
                                    "use angle brackets instead",
                                    format!("{}<{}>", trait_name, args),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        };
                        err.emit();
                        (
                            self.lower_angle_bracketed_parameter_data(
                                &data.as_angle_bracketed_args(),
                                param_mode,
                                itctx
                            ).0,
                            false,
                        )
                    }
                },
            }
        } else {
            self.lower_angle_bracketed_parameter_data(&Default::default(), param_mode, itctx)
        };

        let has_lifetimes = generic_args.args.iter().any(|arg| match arg {
            GenericArg::Lifetime(_) => true,
            _ => false,
        });
        let first_generic_span = generic_args.args.iter().map(|a| a.span())
            .chain(generic_args.bindings.iter().map(|b| b.span)).next();
        if !generic_args.parenthesized && !has_lifetimes {
            generic_args.args =
                self.elided_path_lifetimes(path_span, expected_lifetimes)
                    .into_iter()
                    .map(|lt| GenericArg::Lifetime(lt))
                    .chain(generic_args.args.into_iter())
                .collect();
            if expected_lifetimes > 0 && param_mode == ParamMode::Explicit {
                let anon_lt_suggestion = vec!["'_"; expected_lifetimes].join(", ");
                let no_non_lt_args = generic_args.args.len() == expected_lifetimes;
                let no_bindings = generic_args.bindings.is_empty();
                let (incl_angl_brckt, insertion_sp, suggestion) = if no_non_lt_args && no_bindings {
                    // If there are no (non-implicit) generic args or associated type
                    // bindings, our suggestion includes the angle brackets.
                    (true, path_span.shrink_to_hi(), format!("<{}>", anon_lt_suggestion))
                } else {
                    // Otherwise (sorry, this is kind of gross) we need to infer the
                    // place to splice in the `'_, ` from the generics that do exist.
                    let first_generic_span = first_generic_span
                        .expect("already checked that non-lifetime args or bindings exist");
                    (false, first_generic_span.shrink_to_lo(), format!("{}, ", anon_lt_suggestion))
                };
                match self.anonymous_lifetime_mode {
                    // In create-parameter mode we error here because we don't want to support
                    // deprecated impl elision in new features like impl elision and `async fn`,
                    // both of which work using the `CreateParameter` mode:
                    //
                    //     impl Foo for std::cell::Ref<u32> // note lack of '_
                    //     async fn foo(_: std::cell::Ref<u32>) { ... }
                    AnonymousLifetimeMode::CreateParameter => {
                        let mut err = struct_span_err!(
                            self.sess,
                            path_span,
                            E0726,
                            "implicit elided lifetime not allowed here"
                        );
                        crate::lint::builtin::add_elided_lifetime_in_path_suggestion(
                            &self.sess,
                            &mut err,
                            expected_lifetimes,
                            path_span,
                            incl_angl_brckt,
                            insertion_sp,
                            suggestion,
                        );
                        err.emit();
                    }
                    AnonymousLifetimeMode::PassThrough |
                    AnonymousLifetimeMode::ReportError => {
                        self.sess.buffer_lint_with_diagnostic(
                            ELIDED_LIFETIMES_IN_PATHS,
                            CRATE_NODE_ID,
                            path_span,
                            "hidden lifetime parameters in types are deprecated",
                            builtin::BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                                expected_lifetimes,
                                path_span,
                                incl_angl_brckt,
                                insertion_sp,
                                suggestion,
                            )
                        );
                    }
                }
            }
        }

        let res = self.expect_full_res(segment.id);
        let id = if let Some(owner) = explicit_owner {
            self.lower_node_id_with_owner(segment.id, owner)
        } else {
            self.lower_node_id(segment.id)
        };
        debug!(
            "lower_path_segment: ident={:?} original-id={:?} new-id={:?}",
            segment.ident, segment.id, id,
        );

        hir::PathSegment::new(
            segment.ident,
            Some(id),
            Some(self.lower_res(res)),
            generic_args,
            infer_args,
        )
    }

    fn lower_angle_bracketed_parameter_data(
        &mut self,
        data: &AngleBracketedArgs,
        param_mode: ParamMode,
        mut itctx: ImplTraitContext<'_>,
    ) -> (hir::GenericArgs, bool) {
        let &AngleBracketedArgs { ref args, ref constraints, .. } = data;
        let has_non_lt_args = args.iter().any(|arg| match arg {
            ast::GenericArg::Lifetime(_) => false,
            ast::GenericArg::Type(_) => true,
            ast::GenericArg::Const(_) => true,
        });
        (
            hir::GenericArgs {
                args: args.iter().map(|a| self.lower_generic_arg(a, itctx.reborrow())).collect(),
                bindings: constraints.iter()
                    .map(|b| self.lower_assoc_ty_constraint(b, itctx.reborrow()))
                    .collect(),
                parenthesized: false,
            },
            !has_non_lt_args && param_mode == ParamMode::Optional
        )
    }

    fn lower_parenthesized_parameter_data(
        &mut self,
        data: &ParenthesizedArgs,
    ) -> (hir::GenericArgs, bool) {
        // Switch to `PassThrough` mode for anonymous lifetimes; this
        // means that we permit things like `&Ref<T>`, where `Ref` has
        // a hidden lifetime parameter. This is needed for backwards
        // compatibility, even in contexts like an impl header where
        // we generally don't permit such things (see #51008).
        self.with_anonymous_lifetime_mode(
            AnonymousLifetimeMode::PassThrough,
            |this| {
                let &ParenthesizedArgs { ref inputs, ref output, span } = data;
                let inputs = inputs
                    .iter()
                    .map(|ty| this.lower_ty_direct(ty, ImplTraitContext::disallowed()))
                    .collect();
                let mk_tup = |this: &mut Self, tys, span| {
                    hir::Ty { node: hir::TyKind::Tup(tys), hir_id: this.next_id(), span }
                };
                (
                    hir::GenericArgs {
                        args: hir_vec![GenericArg::Type(mk_tup(this, inputs, span))],
                        bindings: hir_vec![
                            hir::TypeBinding {
                                hir_id: this.next_id(),
                                ident: Ident::with_dummy_span(FN_OUTPUT_NAME),
                                kind: hir::TypeBindingKind::Equality {
                                    ty: output
                                        .as_ref()
                                        .map(|ty| this.lower_ty(
                                            &ty,
                                            ImplTraitContext::disallowed()
                                        ))
                                        .unwrap_or_else(||
                                            P(mk_tup(this, hir::HirVec::new(), span))
                                        ),
                                },
                                span: output.as_ref().map_or(span, |ty| ty.span),
                            }
                        ],
                        parenthesized: true,
                    },
                    false,
                )
            }
        )
    }

    fn lower_local(&mut self, l: &Local) -> (hir::Local, SmallVec<[NodeId; 1]>) {
        let mut ids = SmallVec::<[NodeId; 1]>::new();
        if self.sess.features_untracked().impl_trait_in_bindings {
            if let Some(ref ty) = l.ty {
                let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                visitor.visit_ty(ty);
            }
        }
        let parent_def_id = DefId::local(self.current_hir_id_owner.last().unwrap().0);
        (hir::Local {
            hir_id: self.lower_node_id(l.id),
            ty: l.ty
                .as_ref()
                .map(|t| self.lower_ty(t,
                    if self.sess.features_untracked().impl_trait_in_bindings {
                        ImplTraitContext::OpaqueTy(Some(parent_def_id))
                    } else {
                        ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                    }
                )),
            pat: self.lower_pat(&l.pat),
            init: l.init.as_ref().map(|e| P(self.lower_expr(e))),
            span: l.span,
            attrs: l.attrs.clone(),
            source: hir::LocalSource::Normal,
        }, ids)
    }

    fn lower_mutability(&mut self, m: Mutability) -> hir::Mutability {
        match m {
            Mutability::Mutable => hir::MutMutable,
            Mutability::Immutable => hir::MutImmutable,
        }
    }

    fn lower_fn_params_to_names(&mut self, decl: &FnDecl) -> hir::HirVec<Ident> {
        decl.inputs
            .iter()
            .map(|param| match param.pat.node {
                PatKind::Ident(_, ident, _) => ident,
                _ => Ident::new(kw::Invalid, param.pat.span),
            })
            .collect()
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
        mut in_band_ty_params: Option<(DefId, &mut Vec<hir::GenericParam>)>,
        impl_trait_return_allow: bool,
        make_ret_async: Option<NodeId>,
    ) -> P<hir::FnDecl> {
        let lt_mode = if make_ret_async.is_some() {
            // In `async fn`, argument-position elided lifetimes
            // must be transformed into fresh generic parameters so that
            // they can be applied to the opaque `impl Trait` return type.
            AnonymousLifetimeMode::CreateParameter
        } else {
            self.anonymous_lifetime_mode
        };

        // Remember how many lifetimes were already around so that we can
        // only look at the lifetime parameters introduced by the arguments.
        let inputs = self.with_anonymous_lifetime_mode(lt_mode, |this| {
            decl.inputs
                .iter()
                .map(|param| {
                    if let Some((_, ibty)) = &mut in_band_ty_params {
                        this.lower_ty_direct(&param.ty, ImplTraitContext::Universal(ibty))
                    } else {
                        this.lower_ty_direct(&param.ty, ImplTraitContext::disallowed())
                    }
                })
                .collect::<HirVec<_>>()
        });

        let output = if let Some(ret_id) = make_ret_async {
            self.lower_async_fn_ret_ty(
                &decl.output,
                in_band_ty_params.expect("`make_ret_async` but no `fn_def_id`").0,
                ret_id,
            )
        } else {
            match decl.output {
                FunctionRetTy::Ty(ref ty) => match in_band_ty_params {
                    Some((def_id, _)) if impl_trait_return_allow => {
                        hir::Return(self.lower_ty(ty,
                            ImplTraitContext::OpaqueTy(Some(def_id))
                        ))
                    }
                    _ => {
                        hir::Return(self.lower_ty(ty, ImplTraitContext::disallowed()))
                    }
                },
                FunctionRetTy::Default(span) => hir::DefaultReturn(span),
            }
        };

        P(hir::FnDecl {
            inputs,
            output,
            c_variadic: decl.c_variadic,
            implicit_self: decl.inputs.get(0).map_or(
                hir::ImplicitSelfKind::None,
                |arg| {
                    let is_mutable_pat = match arg.pat.node {
                        PatKind::Ident(BindingMode::ByValue(mt), _, _) |
                        PatKind::Ident(BindingMode::ByRef(mt), _, _) =>
                            mt == Mutability::Mutable,
                        _ => false,
                    };

                    match arg.ty.node {
                        TyKind::ImplicitSelf if is_mutable_pat => hir::ImplicitSelfKind::Mut,
                        TyKind::ImplicitSelf => hir::ImplicitSelfKind::Imm,
                        // Given we are only considering `ImplicitSelf` types, we needn't consider
                        // the case where we have a mutable pattern to a reference as that would
                        // no longer be an `ImplicitSelf`.
                        TyKind::Rptr(_, ref mt) if mt.ty.node.is_implicit_self() &&
                            mt.mutbl == ast::Mutability::Mutable =>
                                hir::ImplicitSelfKind::MutRef,
                        TyKind::Rptr(_, ref mt) if mt.ty.node.is_implicit_self() =>
                            hir::ImplicitSelfKind::ImmRef,
                        _ => hir::ImplicitSelfKind::None,
                    }
                },
            ),
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
    fn lower_async_fn_ret_ty(
        &mut self,
        output: &FunctionRetTy,
        fn_def_id: DefId,
        opaque_ty_node_id: NodeId,
    ) -> hir::FunctionRetTy {
        debug!(
            "lower_async_fn_ret_ty(\
             output={:?}, \
             fn_def_id={:?}, \
             opaque_ty_node_id={:?})",
            output, fn_def_id, opaque_ty_node_id,
        );

        let span = output.span();

        let opaque_ty_span = self.mark_span_with_reason(
            DesugaringKind::Async,
            span,
            None,
        );

        let opaque_ty_def_index = self
            .resolver
            .definitions()
            .opt_def_index(opaque_ty_node_id)
            .unwrap();

        self.allocate_hir_id_counter(opaque_ty_node_id);

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
        //
        // The variable `input_lifetimes_count` tracks the number of
        // lifetime parameters to the opaque type *not counting* those
        // lifetimes elided in the return type. This includes those
        // that are explicitly declared (`in_scope_lifetimes`) and
        // those elided lifetimes we found in the arguments (current
        // content of `lifetimes_to_define`). Next, we will process
        // the return type, which will cause `lifetimes_to_define` to
        // grow.
        let input_lifetimes_count = self.in_scope_lifetimes.len() + self.lifetimes_to_define.len();

        let (opaque_ty_id, lifetime_params) = self.with_hir_id_owner(opaque_ty_node_id, |this| {
            // We have to be careful to get elision right here. The
            // idea is that we create a lifetime parameter for each
            // lifetime in the return type.  So, given a return type
            // like `async fn foo(..) -> &[&u32]`, we lower to `impl
            // Future<Output = &'1 [ &'2 u32 ]>`.
            //
            // Then, we will create `fn foo(..) -> Foo<'_, '_>`, and
            // hence the elision takes place at the fn site.
            let future_bound = this.with_anonymous_lifetime_mode(
                AnonymousLifetimeMode::CreateParameter,
                |this| this.lower_async_fn_output_type_to_future_bound(
                    output,
                    fn_def_id,
                    span,
                ),
            );

            debug!("lower_async_fn_ret_ty: future_bound={:#?}", future_bound);

            // Calculate all the lifetimes that should be captured
            // by the opaque type. This should include all in-scope
            // lifetime parameters, including those defined in-band.
            //
            // Note: this must be done after lowering the output type,
            // as the output type may introduce new in-band lifetimes.
            let lifetime_params: Vec<(Span, ParamName)> =
                this.in_scope_lifetimes
                    .iter().cloned()
                    .map(|name| (name.ident().span, name))
                    .chain(this.lifetimes_to_define.iter().cloned())
                    .collect();

            debug!("lower_async_fn_ret_ty: in_scope_lifetimes={:#?}", this.in_scope_lifetimes);
            debug!("lower_async_fn_ret_ty: lifetimes_to_define={:#?}", this.lifetimes_to_define);
            debug!("lower_async_fn_ret_ty: lifetime_params={:#?}", lifetime_params);

            let generic_params =
                lifetime_params
                    .iter().cloned()
                    .map(|(span, hir_name)| {
                        this.lifetime_to_generic_param(span, hir_name, opaque_ty_def_index)
                    })
                    .collect();

            let opaque_ty_item = hir::OpaqueTy {
                generics: hir::Generics {
                    params: generic_params,
                    where_clause: hir::WhereClause {
                        predicates: hir_vec![],
                        span,
                    },
                    span,
                },
                bounds: hir_vec![future_bound],
                impl_trait_fn: Some(fn_def_id),
                origin: hir::OpaqueTyOrigin::AsyncFn,
            };

            trace!("exist ty from async fn def index: {:#?}", opaque_ty_def_index);
            let opaque_ty_id = this.generate_opaque_type(
                opaque_ty_node_id,
                opaque_ty_item,
                span,
                opaque_ty_span,
            );

            (opaque_ty_id, lifetime_params)
        });

        // As documented above on the variable
        // `input_lifetimes_count`, we need to create the lifetime
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
        let mut generic_args: Vec<_> =
            lifetime_params[..input_lifetimes_count]
            .iter()
            .map(|&(span, hir_name)| {
                // Input lifetime like `'a` or `'1`:
                GenericArg::Lifetime(hir::Lifetime {
                    hir_id: self.next_id(),
                    span,
                    name: hir::LifetimeName::Param(hir_name),
                })
            })
            .collect();
        generic_args.extend(
            lifetime_params[input_lifetimes_count..]
            .iter()
            .map(|&(span, _)| {
                // Output lifetime like `'_`.
                GenericArg::Lifetime(hir::Lifetime {
                    hir_id: self.next_id(),
                    span,
                    name: hir::LifetimeName::Implicit,
                })
            })
        );

        // Create the `Foo<...>` refernece itself. Note that the `type
        // Foo = impl Trait` is, internally, created as a child of the
        // async fn, so the *type parameters* are inherited.  It's
        // only the lifetime parameters that we must supply.
        let opaque_ty_ref = hir::TyKind::Def(hir::ItemId { id: opaque_ty_id }, generic_args.into());

        hir::FunctionRetTy::Return(P(hir::Ty {
            node: opaque_ty_ref,
            span,
            hir_id: self.next_id(),
        }))
    }

    /// Transforms `-> T` into `Future<Output = T>`
    fn lower_async_fn_output_type_to_future_bound(
        &mut self,
        output: &FunctionRetTy,
        fn_def_id: DefId,
        span: Span,
    ) -> hir::GenericBound {
        // Compute the `T` in `Future<Output = T>` from the return type.
        let output_ty = match output {
            FunctionRetTy::Ty(ty) => {
                self.lower_ty(ty, ImplTraitContext::OpaqueTy(Some(fn_def_id)))
            }
            FunctionRetTy::Default(ret_ty_span) => {
                P(hir::Ty {
                    hir_id: self.next_id(),
                    node: hir::TyKind::Tup(hir_vec![]),
                    span: *ret_ty_span,
                })
            }
        };

        // "<Output = T>"
        let future_params = P(hir::GenericArgs {
            args: hir_vec![],
            bindings: hir_vec![hir::TypeBinding {
                ident: Ident::with_dummy_span(FN_OUTPUT_NAME),
                kind: hir::TypeBindingKind::Equality {
                    ty: output_ty,
                },
                hir_id: self.next_id(),
                span,
            }],
            parenthesized: false,
        });

        // ::std::future::Future<future_params>
        let future_path =
            P(self.std_path(span, &[sym::future, sym::Future], Some(future_params), false));

        hir::GenericBound::Trait(
            hir::PolyTraitRef {
                trait_ref: hir::TraitRef {
                    path: future_path,
                    hir_ref_id: self.next_id(),
                },
                bound_generic_params: hir_vec![],
                span,
            },
            hir::TraitBoundModifier::None,
        )
    }

    fn lower_param_bound(
        &mut self,
        tpb: &GenericBound,
        itctx: ImplTraitContext<'_>,
    ) -> hir::GenericBound {
        match *tpb {
            GenericBound::Trait(ref ty, modifier) => {
                hir::GenericBound::Trait(
                    self.lower_poly_trait_ref(ty, itctx),
                    self.lower_trait_bound_modifier(modifier),
                )
            }
            GenericBound::Outlives(ref lifetime) => {
                hir::GenericBound::Outlives(self.lower_lifetime(lifetime))
            }
        }
    }

    fn lower_lifetime(&mut self, l: &Lifetime) -> hir::Lifetime {
        let span = l.ident.span;
        match l.ident {
            ident if ident.name == kw::StaticLifetime =>
                self.new_named_lifetime(l.id, span, hir::LifetimeName::Static),
            ident if ident.name == kw::UnderscoreLifetime =>
                match self.anonymous_lifetime_mode {
                    AnonymousLifetimeMode::CreateParameter => {
                        let fresh_name = self.collect_fresh_in_band_lifetime(span);
                        self.new_named_lifetime(l.id, span, hir::LifetimeName::Param(fresh_name))
                    }

                    AnonymousLifetimeMode::PassThrough => {
                        self.new_named_lifetime(l.id, span, hir::LifetimeName::Underscore)
                    }

                    AnonymousLifetimeMode::ReportError => self.new_error_lifetime(Some(l.id), span),
                },
            ident => {
                self.maybe_collect_in_band_lifetime(ident);
                let param_name = ParamName::Plain(ident);
                self.new_named_lifetime(l.id, span, hir::LifetimeName::Param(param_name))
            }
        }
    }

    fn new_named_lifetime(
        &mut self,
        id: NodeId,
        span: Span,
        name: hir::LifetimeName,
    ) -> hir::Lifetime {
        hir::Lifetime {
            hir_id: self.lower_node_id(id),
            span,
            name: name,
        }
    }

    fn lower_generic_params(
        &mut self,
        params: &[GenericParam],
        add_bounds: &NodeMap<Vec<GenericBound>>,
        mut itctx: ImplTraitContext<'_>,
    ) -> hir::HirVec<hir::GenericParam> {
        params.iter().map(|param| {
            self.lower_generic_param(param, add_bounds, itctx.reborrow())
        }).collect()
    }

    fn lower_generic_param(&mut self,
                           param: &GenericParam,
                           add_bounds: &NodeMap<Vec<GenericBound>>,
                           mut itctx: ImplTraitContext<'_>)
                           -> hir::GenericParam {
        let mut bounds = self.with_anonymous_lifetime_mode(
            AnonymousLifetimeMode::ReportError,
            |this| this.lower_param_bounds(&param.bounds, itctx.reborrow()),
        );

        let (name, kind) = match param.kind {
            GenericParamKind::Lifetime => {
                let was_collecting_in_band = self.is_collecting_in_band_lifetimes;
                self.is_collecting_in_band_lifetimes = false;

                let lt = self.with_anonymous_lifetime_mode(
                    AnonymousLifetimeMode::ReportError,
                    |this| this.lower_lifetime(&Lifetime { id: param.id, ident: param.ident }),
                );
                let param_name = match lt.name {
                    hir::LifetimeName::Param(param_name) => param_name,
                    hir::LifetimeName::Implicit
                        | hir::LifetimeName::Underscore
                        | hir::LifetimeName::Static => hir::ParamName::Plain(lt.name.ident()),
                    hir::LifetimeName::ImplicitObjectLifetimeDefault => {
                        span_bug!(
                            param.ident.span,
                            "object-lifetime-default should not occur here",
                        );
                    }
                    hir::LifetimeName::Error => ParamName::Error,
                };

                let kind = hir::GenericParamKind::Lifetime {
                    kind: hir::LifetimeParamKind::Explicit
                };

                self.is_collecting_in_band_lifetimes = was_collecting_in_band;

                (param_name, kind)
            }
            GenericParamKind::Type { ref default, .. } => {
                let add_bounds = add_bounds.get(&param.id).map_or(&[][..], |x| &x);
                if !add_bounds.is_empty() {
                    let params = self.lower_param_bounds(add_bounds, itctx.reborrow()).into_iter();
                    bounds = bounds.into_iter()
                                   .chain(params)
                                   .collect();
                }

                let kind = hir::GenericParamKind::Type {
                    default: default.as_ref().map(|x| {
                        self.lower_ty(x, ImplTraitContext::OpaqueTy(None))
                    }),
                    synthetic: param.attrs.iter()
                                          .filter(|attr| attr.check_name(sym::rustc_synthetic))
                                          .map(|_| hir::SyntheticTyParamKind::ImplTrait)
                                          .next(),
                };

                (hir::ParamName::Plain(param.ident), kind)
            }
            GenericParamKind::Const { ref ty } => {
                (hir::ParamName::Plain(param.ident), hir::GenericParamKind::Const {
                    ty: self.lower_ty(&ty, ImplTraitContext::disallowed()),
                })
            }
        };

        hir::GenericParam {
            hir_id: self.lower_node_id(param.id),
            name,
            span: param.ident.span,
            pure_wrt_drop: attr::contains_name(&param.attrs, sym::may_dangle),
            attrs: self.lower_attrs(&param.attrs),
            bounds,
            kind,
        }
    }

    fn lower_trait_ref(&mut self, p: &TraitRef, itctx: ImplTraitContext<'_>) -> hir::TraitRef {
        let path = match self.lower_qpath(p.ref_id, &None, &p.path, ParamMode::Explicit, itctx) {
            hir::QPath::Resolved(None, path) => path,
            qpath => bug!("lower_trait_ref: unexpected QPath `{:?}`", qpath),
        };
        hir::TraitRef {
            path,
            hir_ref_id: self.lower_node_id(p.ref_id),
        }
    }

    fn lower_poly_trait_ref(
        &mut self,
        p: &PolyTraitRef,
        mut itctx: ImplTraitContext<'_>,
    ) -> hir::PolyTraitRef {
        let bound_generic_params = self.lower_generic_params(
            &p.bound_generic_params,
            &NodeMap::default(),
            itctx.reborrow(),
        );
        let trait_ref = self.with_in_scope_lifetime_defs(
            &p.bound_generic_params,
            |this| this.lower_trait_ref(&p.trait_ref, itctx),
        );

        hir::PolyTraitRef {
            bound_generic_params,
            trait_ref,
            span: p.span,
        }
    }

    fn lower_mt(&mut self, mt: &MutTy, itctx: ImplTraitContext<'_>) -> hir::MutTy {
        hir::MutTy {
            ty: self.lower_ty(&mt.ty, itctx),
            mutbl: self.lower_mutability(mt.mutbl),
        }
    }

    fn lower_param_bounds(&mut self, bounds: &[GenericBound], mut itctx: ImplTraitContext<'_>)
                          -> hir::GenericBounds {
        bounds.iter().map(|bound| self.lower_param_bound(bound, itctx.reborrow())).collect()
    }

    fn lower_block(&mut self, b: &Block, targeted_by_break: bool) -> P<hir::Block> {
        let mut stmts = vec![];
        let mut expr = None;

        for (index, stmt) in b.stmts.iter().enumerate() {
            if index == b.stmts.len() - 1 {
                if let StmtKind::Expr(ref e) = stmt.node {
                    expr = Some(P(self.lower_expr(e)));
                } else {
                    stmts.extend(self.lower_stmt(stmt));
                }
            } else {
                stmts.extend(self.lower_stmt(stmt));
            }
        }

        P(hir::Block {
            hir_id: self.lower_node_id(b.id),
            stmts: stmts.into(),
            expr,
            rules: self.lower_block_check_mode(&b.rules),
            span: b.span,
            targeted_by_break,
        })
    }

    /// Lowers a block directly to an expression, presuming that it
    /// has no attributes and is not targeted by a `break`.
    fn lower_block_expr(&mut self, b: &Block) -> hir::Expr {
        let block = self.lower_block(b, false);
        self.expr_block(block, ThinVec::new())
    }

    fn lower_pat(&mut self, p: &Pat) -> P<hir::Pat> {
        let node = match p.node {
            PatKind::Wild => hir::PatKind::Wild,
            PatKind::Ident(ref binding_mode, ident, ref sub) => {
                let lower_sub = |this: &mut Self| sub.as_ref().map(|x| this.lower_pat(x));
                self.lower_pat_ident(p, binding_mode, ident, lower_sub)
            }
            PatKind::Lit(ref e) => hir::PatKind::Lit(P(self.lower_expr(e))),
            PatKind::TupleStruct(ref path, ref pats) => {
                let qpath = self.lower_qpath(
                    p.id,
                    &None,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );
                let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple struct");
                hir::PatKind::TupleStruct(qpath, pats, ddpos)
            }
            PatKind::Or(ref pats) => {
                hir::PatKind::Or(pats.iter().map(|x| self.lower_pat(x)).collect())
            }
            PatKind::Path(ref qself, ref path) => {
                let qpath = self.lower_qpath(
                    p.id,
                    qself,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );
                hir::PatKind::Path(qpath)
            }
            PatKind::Struct(ref path, ref fields, etc) => {
                let qpath = self.lower_qpath(
                    p.id,
                    &None,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );

                let fs = fields
                    .iter()
                    .map(|f| hir::FieldPat {
                        hir_id: self.next_id(),
                        ident: f.ident,
                        pat: self.lower_pat(&f.pat),
                        is_shorthand: f.is_shorthand,
                        span: f.span,
                    })
                    .collect();
                hir::PatKind::Struct(qpath, fs, etc)
            }
            PatKind::Tuple(ref pats) => {
                let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple");
                hir::PatKind::Tuple(pats, ddpos)
            }
            PatKind::Box(ref inner) => hir::PatKind::Box(self.lower_pat(inner)),
            PatKind::Ref(ref inner, mutbl) => {
                hir::PatKind::Ref(self.lower_pat(inner), self.lower_mutability(mutbl))
            }
            PatKind::Range(ref e1, ref e2, Spanned { node: ref end, .. }) => hir::PatKind::Range(
                P(self.lower_expr(e1)),
                P(self.lower_expr(e2)),
                self.lower_range_end(end),
            ),
            PatKind::Slice(ref pats) => self.lower_pat_slice(pats),
            PatKind::Rest => {
                // If we reach here the `..` pattern is not semantically allowed.
                self.ban_illegal_rest_pat(p.span)
            }
            PatKind::Paren(ref inner) => return self.lower_pat(inner),
            PatKind::Mac(_) => panic!("Shouldn't exist here"),
        };

        self.pat_with_node_id_of(p, node)
    }

    fn lower_pat_tuple(
        &mut self,
        pats: &[AstP<Pat>],
        ctx: &str,
    ) -> (HirVec<P<hir::Pat>>, Option<usize>) {
        let mut elems = Vec::with_capacity(pats.len());
        let mut rest = None;

        let mut iter = pats.iter().enumerate();
        while let Some((idx, pat)) = iter.next() {
            // Interpret the first `..` pattern as a subtuple pattern.
            if pat.is_rest() {
                rest = Some((idx, pat.span));
                break;
            }
            // It was not a subslice pattern so lower it normally.
            elems.push(self.lower_pat(pat));
        }

        while let Some((_, pat)) = iter.next() {
            // There was a previous subtuple pattern; make sure we don't allow more.
            if pat.is_rest() {
                self.ban_extra_rest_pat(pat.span, rest.unwrap().1, ctx);
            } else {
                elems.push(self.lower_pat(pat));
            }
        }

        (elems.into(), rest.map(|(ddpos, _)| ddpos))
    }

    fn lower_pat_slice(&mut self, pats: &[AstP<Pat>]) -> hir::PatKind {
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut slice = None;
        let mut prev_rest_span = None;

        let mut iter = pats.iter();
        while let Some(pat) = iter.next() {
            // Interpret the first `((ref mut?)? x @)? ..` pattern as a subslice pattern.
            match pat.node {
                PatKind::Rest => {
                    prev_rest_span = Some(pat.span);
                    slice = Some(self.pat_wild_with_node_id_of(pat));
                    break;
                },
                PatKind::Ident(ref bm, ident, Some(ref sub)) if sub.is_rest() => {
                    prev_rest_span = Some(sub.span);
                    let lower_sub = |this: &mut Self| Some(this.pat_wild_with_node_id_of(sub));
                    let node = self.lower_pat_ident(pat, bm, ident, lower_sub);
                    slice = Some(self.pat_with_node_id_of(pat, node));
                    break;
                },
                _ => {}
            }

            // It was not a subslice pattern so lower it normally.
            before.push(self.lower_pat(pat));
        }

        while let Some(pat) = iter.next() {
            // There was a previous subslice pattern; make sure we don't allow more.
            let rest_span = match pat.node {
                PatKind::Rest => Some(pat.span),
                PatKind::Ident(.., Some(ref sub)) if sub.is_rest() => {
                    // The `HirValidator` is merciless; add a `_` pattern to avoid ICEs.
                    after.push(self.pat_wild_with_node_id_of(pat));
                    Some(sub.span)
                },
                _ => None,
            };
            if let Some(rest_span) = rest_span {
                self.ban_extra_rest_pat(rest_span, prev_rest_span.unwrap(), "slice");
            } else {
                after.push(self.lower_pat(pat));
            }
        }

        hir::PatKind::Slice(before.into(), slice, after.into())
    }

    fn lower_pat_ident(
        &mut self,
        p: &Pat,
        binding_mode: &BindingMode,
        ident: Ident,
        lower_sub: impl FnOnce(&mut Self) -> Option<P<hir::Pat>>,
    ) -> hir::PatKind {
        match self.resolver.get_partial_res(p.id).map(|d| d.base_res()) {
            // `None` can occur in body-less function signatures
            res @ None | res @ Some(Res::Local(_)) => {
                let canonical_id = match res {
                    Some(Res::Local(id)) => id,
                    _ => p.id,
                };

                hir::PatKind::Binding(
                    self.lower_binding_mode(binding_mode),
                    self.lower_node_id(canonical_id),
                    ident,
                    lower_sub(self),
                )
            }
            Some(res) => hir::PatKind::Path(hir::QPath::Resolved(
                None,
                P(hir::Path {
                    span: ident.span,
                    res: self.lower_res(res),
                    segments: hir_vec![hir::PathSegment::from_ident(ident)],
                }),
            )),
        }
    }

    fn pat_wild_with_node_id_of(&mut self, p: &Pat) -> P<hir::Pat> {
        self.pat_with_node_id_of(p, hir::PatKind::Wild)
    }

    /// Construct a `Pat` with the `HirId` of `p.id` lowered.
    fn pat_with_node_id_of(&mut self, p: &Pat, node: hir::PatKind) -> P<hir::Pat> {
        P(hir::Pat {
            hir_id: self.lower_node_id(p.id),
            node,
            span: p.span,
        })
    }

    /// Emit a friendly error for extra `..` patterns in a tuple/tuple struct/slice pattern.
    fn ban_extra_rest_pat(&self, sp: Span, prev_sp: Span, ctx: &str) {
        self.diagnostic()
            .struct_span_err(sp, &format!("`..` can only be used once per {} pattern", ctx))
            .span_label(sp, &format!("can only be used once per {} pattern", ctx))
            .span_label(prev_sp, "previously used here")
            .emit();
    }

    /// Used to ban the `..` pattern in places it shouldn't be semantically.
    fn ban_illegal_rest_pat(&self, sp: Span) -> hir::PatKind {
        self.diagnostic()
            .struct_span_err(sp, "`..` patterns are not allowed here")
            .note("only allowed in tuple, tuple struct, and slice patterns")
            .emit();

        // We're not in a list context so `..` can be reasonably treated
        // as `_` because it should always be valid and roughly matches the
        // intent of `..` (notice that the rest of a single slot is that slot).
        hir::PatKind::Wild
    }

    fn lower_range_end(&mut self, e: &RangeEnd) -> hir::RangeEnd {
        match *e {
            RangeEnd::Included(_) => hir::RangeEnd::Included,
            RangeEnd::Excluded => hir::RangeEnd::Excluded,
        }
    }

    fn lower_anon_const(&mut self, c: &AnonConst) -> hir::AnonConst {
        self.with_new_scopes(|this| {
            hir::AnonConst {
                hir_id: this.lower_node_id(c.id),
                body: this.lower_const_body(&c.value),
            }
        })
    }

    fn lower_stmt(&mut self, s: &Stmt) -> SmallVec<[hir::Stmt; 1]> {
        let node = match s.node {
            StmtKind::Local(ref l) => {
                let (l, item_ids) = self.lower_local(l);
                let mut ids: SmallVec<[hir::Stmt; 1]> = item_ids
                    .into_iter()
                    .map(|item_id| {
                        let item_id = hir::ItemId { id: self.lower_node_id(item_id) };
                        self.stmt(s.span, hir::StmtKind::Item(item_id))
                    })
                    .collect();
                ids.push({
                    hir::Stmt {
                        hir_id: self.lower_node_id(s.id),
                        node: hir::StmtKind::Local(P(l)),
                        span: s.span,
                    }
                });
                return ids;
            },
            StmtKind::Item(ref it) => {
                // Can only use the ID once.
                let mut id = Some(s.id);
                return self.lower_item_id(it)
                    .into_iter()
                    .map(|item_id| {
                        let hir_id = id.take()
                          .map(|id| self.lower_node_id(id))
                          .unwrap_or_else(|| self.next_id());

                        hir::Stmt {
                            hir_id,
                            node: hir::StmtKind::Item(item_id),
                            span: s.span,
                        }
                    })
                    .collect();
            }
            StmtKind::Expr(ref e) => hir::StmtKind::Expr(P(self.lower_expr(e))),
            StmtKind::Semi(ref e) => hir::StmtKind::Semi(P(self.lower_expr(e))),
            StmtKind::Mac(..) => panic!("shouldn't exist here"),
        };
        smallvec![hir::Stmt {
            hir_id: self.lower_node_id(s.id),
            node,
            span: s.span,
        }]
    }

    fn lower_block_check_mode(&mut self, b: &BlockCheckMode) -> hir::BlockCheckMode {
        match *b {
            BlockCheckMode::Default => hir::DefaultBlock,
            BlockCheckMode::Unsafe(u) => hir::UnsafeBlock(self.lower_unsafe_source(u)),
        }
    }

    fn lower_binding_mode(&mut self, b: &BindingMode) -> hir::BindingAnnotation {
        match *b {
            BindingMode::ByValue(Mutability::Immutable) => hir::BindingAnnotation::Unannotated,
            BindingMode::ByRef(Mutability::Immutable) => hir::BindingAnnotation::Ref,
            BindingMode::ByValue(Mutability::Mutable) => hir::BindingAnnotation::Mutable,
            BindingMode::ByRef(Mutability::Mutable) => hir::BindingAnnotation::RefMut,
        }
    }

    fn lower_unsafe_source(&mut self, u: UnsafeSource) -> hir::UnsafeSource {
        match u {
            CompilerGenerated => hir::CompilerGenerated,
            UserProvided => hir::UserProvided,
        }
    }

    fn lower_trait_bound_modifier(&mut self, f: TraitBoundModifier) -> hir::TraitBoundModifier {
        match f {
            TraitBoundModifier::None => hir::TraitBoundModifier::None,
            TraitBoundModifier::Maybe => hir::TraitBoundModifier::Maybe,
        }
    }

    // Helper methods for building HIR.

    fn stmt(&mut self, span: Span, node: hir::StmtKind) -> hir::Stmt {
        hir::Stmt { span, node, hir_id: self.next_id() }
    }

    fn stmt_expr(&mut self, span: Span, expr: hir::Expr) -> hir::Stmt {
        self.stmt(span, hir::StmtKind::Expr(P(expr)))
    }

    fn stmt_let_pat(
        &mut self,
        attrs: ThinVec<Attribute>,
        span: Span,
        init: Option<P<hir::Expr>>,
        pat: P<hir::Pat>,
        source: hir::LocalSource,
    ) -> hir::Stmt {
        let local = hir::Local {
            attrs,
            hir_id: self.next_id(),
            init,
            pat,
            source,
            span,
            ty: None,
        };
        self.stmt(span, hir::StmtKind::Local(P(local)))
    }

    fn block_expr(&mut self, expr: P<hir::Expr>) -> hir::Block {
        self.block_all(expr.span, hir::HirVec::new(), Some(expr))
    }

    fn block_all(
        &mut self,
        span: Span,
        stmts: hir::HirVec<hir::Stmt>,
        expr: Option<P<hir::Expr>>,
    ) -> hir::Block {
        hir::Block {
            stmts,
            expr,
            hir_id: self.next_id(),
            rules: hir::DefaultBlock,
            span,
            targeted_by_break: false,
        }
    }

    /// Constructs a `true` or `false` literal pattern.
    fn pat_bool(&mut self, span: Span, val: bool) -> P<hir::Pat> {
        let expr = self.expr_bool(span, val);
        self.pat(span, hir::PatKind::Lit(P(expr)))
    }

    fn pat_ok(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &[sym::result, sym::Result, sym::Ok], hir_vec![pat])
    }

    fn pat_err(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &[sym::result, sym::Result, sym::Err], hir_vec![pat])
    }

    fn pat_some(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &[sym::option, sym::Option, sym::Some], hir_vec![pat])
    }

    fn pat_none(&mut self, span: Span) -> P<hir::Pat> {
        self.pat_std_enum(span, &[sym::option, sym::Option, sym::None], hir_vec![])
    }

    fn pat_std_enum(
        &mut self,
        span: Span,
        components: &[Symbol],
        subpats: hir::HirVec<P<hir::Pat>>,
    ) -> P<hir::Pat> {
        let path = self.std_path(span, components, None, true);
        let qpath = hir::QPath::Resolved(None, P(path));
        let pt = if subpats.is_empty() {
            hir::PatKind::Path(qpath)
        } else {
            hir::PatKind::TupleStruct(qpath, subpats, None)
        };
        self.pat(span, pt)
    }

    fn pat_ident(&mut self, span: Span, ident: Ident) -> (P<hir::Pat>, hir::HirId) {
        self.pat_ident_binding_mode(span, ident, hir::BindingAnnotation::Unannotated)
    }

    fn pat_ident_binding_mode(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> (P<hir::Pat>, hir::HirId) {
        let hir_id = self.next_id();

        (
            P(hir::Pat {
                hir_id,
                node: hir::PatKind::Binding(bm, hir_id, ident.with_span_pos(span), None),
                span,
            }),
            hir_id
        )
    }

    fn pat_wild(&mut self, span: Span) -> P<hir::Pat> {
        self.pat(span, hir::PatKind::Wild)
    }

    fn pat(&mut self, span: Span, pat: hir::PatKind) -> P<hir::Pat> {
        P(hir::Pat {
            hir_id: self.next_id(),
            node: pat,
            span,
        })
    }

    /// Given a suffix `["b", "c", "d"]`, returns path `::std::b::c::d` when
    /// `fld.cx.use_std`, and `::core::b::c::d` otherwise.
    /// The path is also resolved according to `is_value`.
    fn std_path(
        &mut self,
        span: Span,
        components: &[Symbol],
        params: Option<P<hir::GenericArgs>>,
        is_value: bool,
    ) -> hir::Path {
        let ns = if is_value { Namespace::ValueNS } else { Namespace::TypeNS };
        let (path, res) = self.resolver.resolve_str_path(span, self.crate_root, components, ns);

        let mut segments: Vec<_> = path.segments.iter().map(|segment| {
            let res = self.expect_full_res(segment.id);
            hir::PathSegment {
                ident: segment.ident,
                hir_id: Some(self.lower_node_id(segment.id)),
                res: Some(self.lower_res(res)),
                infer_args: true,
                args: None,
            }
        }).collect();
        segments.last_mut().unwrap().args = params;

        hir::Path {
            span,
            res: res.map_id(|_| panic!("unexpected `NodeId`")),
            segments: segments.into(),
        }
    }

    fn ty_path(&mut self, mut hir_id: hir::HirId, span: Span, qpath: hir::QPath) -> hir::Ty {
        let node = match qpath {
            hir::QPath::Resolved(None, path) => {
                // Turn trait object paths into `TyKind::TraitObject` instead.
                match path.res {
                    Res::Def(DefKind::Trait, _) | Res::Def(DefKind::TraitAlias, _) => {
                        let principal = hir::PolyTraitRef {
                            bound_generic_params: hir::HirVec::new(),
                            trait_ref: hir::TraitRef {
                                path,
                                hir_ref_id: hir_id,
                            },
                            span,
                        };

                        // The original ID is taken by the `PolyTraitRef`,
                        // so the `Ty` itself needs a different one.
                        hir_id = self.next_id();
                        hir::TyKind::TraitObject(hir_vec![principal], self.elided_dyn_bound(span))
                    }
                    _ => hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                }
            }
            _ => hir::TyKind::Path(qpath),
        };
        hir::Ty {
            hir_id,
            node,
            span,
        }
    }

    /// Invoked to create the lifetime argument for a type `&T`
    /// with no explicit lifetime.
    fn elided_ref_lifetime(&mut self, span: Span) -> hir::Lifetime {
        match self.anonymous_lifetime_mode {
            // Intercept when we are in an impl header or async fn and introduce an in-band
            // lifetime.
            // Hence `impl Foo for &u32` becomes `impl<'f> Foo for &'f u32` for some fresh
            // `'f`.
            AnonymousLifetimeMode::CreateParameter => {
                let fresh_name = self.collect_fresh_in_band_lifetime(span);
                hir::Lifetime {
                    hir_id: self.next_id(),
                    span,
                    name: hir::LifetimeName::Param(fresh_name),
                }
            }

            AnonymousLifetimeMode::ReportError => self.new_error_lifetime(None, span),

            AnonymousLifetimeMode::PassThrough => self.new_implicit_lifetime(span),
        }
    }

    /// Report an error on illegal use of `'_` or a `&T` with no explicit lifetime;
    /// return a "error lifetime".
    fn new_error_lifetime(&mut self, id: Option<NodeId>, span: Span) -> hir::Lifetime {
        let (id, msg, label) = match id {
            Some(id) => (id, "`'_` cannot be used here", "`'_` is a reserved lifetime name"),

            None => (
                self.sess.next_node_id(),
                "`&` without an explicit lifetime name cannot be used here",
                "explicit lifetime name needed here",
            ),
        };

        let mut err = struct_span_err!(
            self.sess,
            span,
            E0637,
            "{}",
            msg,
        );
        err.span_label(span, label);
        err.emit();

        self.new_named_lifetime(id, span, hir::LifetimeName::Error)
    }

    /// Invoked to create the lifetime argument(s) for a path like
    /// `std::cell::Ref<T>`; note that implicit lifetimes in these
    /// sorts of cases are deprecated. This may therefore report a warning or an
    /// error, depending on the mode.
    fn elided_path_lifetimes(&mut self, span: Span, count: usize) -> P<[hir::Lifetime]> {
        (0..count)
            .map(|_| self.elided_path_lifetime(span))
            .collect()
    }

    fn elided_path_lifetime(&mut self, span: Span) -> hir::Lifetime {
        match self.anonymous_lifetime_mode {
            AnonymousLifetimeMode::CreateParameter => {
                // We should have emitted E0726 when processing this path above
                self.sess.delay_span_bug(
                    span,
                    "expected 'implicit elided lifetime not allowed' error",
                );
                let id = self.sess.next_node_id();
                self.new_named_lifetime(id, span, hir::LifetimeName::Error)
            }
            // This is the normal case.
            AnonymousLifetimeMode::PassThrough => self.new_implicit_lifetime(span),

            AnonymousLifetimeMode::ReportError => self.new_error_lifetime(None, span),
        }
    }

    /// Invoked to create the lifetime argument(s) for an elided trait object
    /// bound, like the bound in `Box<dyn Debug>`. This method is not invoked
    /// when the bound is written, even if it is written with `'_` like in
    /// `Box<dyn Debug + '_>`. In those cases, `lower_lifetime` is invoked.
    fn elided_dyn_bound(&mut self, span: Span) -> hir::Lifetime {
        match self.anonymous_lifetime_mode {
            // NB. We intentionally ignore the create-parameter mode here.
            // and instead "pass through" to resolve-lifetimes, which will apply
            // the object-lifetime-defaulting rules. Elided object lifetime defaults
            // do not act like other elided lifetimes. In other words, given this:
            //
            //     impl Foo for Box<dyn Debug>
            //
            // we do not introduce a fresh `'_` to serve as the bound, but instead
            // ultimately translate to the equivalent of:
            //
            //     impl Foo for Box<dyn Debug + 'static>
            //
            // `resolve_lifetime` has the code to make that happen.
            AnonymousLifetimeMode::CreateParameter => {}

            AnonymousLifetimeMode::ReportError => {
                // ReportError applies to explicit use of `'_`.
            }

            // This is the normal case.
            AnonymousLifetimeMode::PassThrough => {}
        }

        let r = hir::Lifetime {
            hir_id: self.next_id(),
            span,
            name: hir::LifetimeName::ImplicitObjectLifetimeDefault,
        };
        debug!("elided_dyn_bound: r={:?}", r);
        r
    }

    fn new_implicit_lifetime(&mut self, span: Span) -> hir::Lifetime {
        hir::Lifetime {
            hir_id: self.next_id(),
            span,
            name: hir::LifetimeName::Implicit,
        }
    }

    fn maybe_lint_bare_trait(&self, span: Span, id: NodeId, is_global: bool) {
        // FIXME(davidtwco): This is a hack to detect macros which produce spans of the
        // call site which do not have a macro backtrace. See #61963.
        let is_macro_callsite = self.sess.source_map()
            .span_to_snippet(span)
            .map(|snippet| snippet.starts_with("#["))
            .unwrap_or(true);
        if !is_macro_callsite {
            self.sess.buffer_lint_with_diagnostic(
                builtin::BARE_TRAIT_OBJECTS,
                id,
                span,
                "trait objects without an explicit `dyn` are deprecated",
                builtin::BuiltinLintDiagnostics::BareTraitObject(span, is_global),
            )
        }
    }
}

fn body_ids(bodies: &BTreeMap<hir::BodyId, hir::Body>) -> Vec<hir::BodyId> {
    // Sorting by span ensures that we get things in order within a
    // file, and also puts the files in a sensible order.
    let mut body_ids: Vec<_> = bodies.keys().cloned().collect();
    body_ids.sort_by_key(|b| bodies[b].value.span);
    body_ids
}

/// Checks if the specified expression is a built-in range literal.
/// (See: `LoweringContext::lower_expr()`).
pub fn is_range_literal(sess: &Session, expr: &hir::Expr) -> bool {
    use hir::{Path, QPath, ExprKind, TyKind};

    // Returns whether the given path represents a (desugared) range,
    // either in std or core, i.e. has either a `::std::ops::Range` or
    // `::core::ops::Range` prefix.
    fn is_range_path(path: &Path) -> bool {
        let segs: Vec<_> = path.segments.iter().map(|seg| seg.ident.as_str().to_string()).collect();
        let segs: Vec<_> = segs.iter().map(|seg| &**seg).collect();

        // "{{root}}" is the equivalent of `::` prefix in `Path`.
        if let ["{{root}}", std_core, "ops", range] = segs.as_slice() {
            (*std_core == "std" || *std_core == "core") && range.starts_with("Range")
        } else {
            false
        }
    };

    // Check whether a span corresponding to a range expression is a
    // range literal, rather than an explicit struct or `new()` call.
    fn is_lit(sess: &Session, span: &Span) -> bool {
        let source_map = sess.source_map();
        let end_point = source_map.end_point(*span);

        if let Ok(end_string) = source_map.span_to_snippet(end_point) {
            !(end_string.ends_with("}") || end_string.ends_with(")"))
        } else {
            false
        }
    };

    match expr.node {
        // All built-in range literals but `..=` and `..` desugar to `Struct`s.
        ExprKind::Struct(ref qpath, _, _) => {
            if let QPath::Resolved(None, ref path) = **qpath {
                return is_range_path(&path) && is_lit(sess, &expr.span);
            }
        }

        // `..` desugars to its struct path.
        ExprKind::Path(QPath::Resolved(None, ref path)) => {
            return is_range_path(&path) && is_lit(sess, &expr.span);
        }

        // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
        ExprKind::Call(ref func, _) => {
            if let ExprKind::Path(QPath::TypeRelative(ref ty, ref segment)) = func.node {
                if let TyKind::Path(QPath::Resolved(None, ref path)) = ty.node {
                    let new_call = segment.ident.as_str() == "new";
                    return is_range_path(&path) && is_lit(sess, &expr.span) && new_call;
                }
            }
        }

        _ => {}
    }

    false
}
