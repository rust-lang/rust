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

use crate::dep_graph::DepGraph;
use crate::hir::{self, ParamName};
use crate::hir::HirVec;
use crate::hir::map::{DefKey, DefPathData, Definitions};
use crate::hir::def_id::{DefId, DefIndex, CRATE_DEF_INDEX};
use crate::hir::def::{Res, DefKind, PartialRes, PerNS};
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

use std::collections::{BTreeSet, BTreeMap};
use std::mem;
use smallvec::SmallVec;
use syntax::attr;
use syntax::ast;
use syntax::ast::*;
use syntax::errors;
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::print::pprust;
use syntax::source_map::{self, respan, ExpnInfo, CompilerDesugaringKind, Spanned};
use syntax::source_map::CompilerDesugaringKind::IfTemporary;
use syntax::std_inject;
use syntax::symbol::{kw, sym, Symbol};
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax::parse::token::{self, Token};
use syntax::visit::{self, Visitor};
use syntax_pos::{DUMMY_SP, Span};

const HIR_ID_COUNTER_LOCKED: u32 = 0xFFFFFFFF;

pub struct LoweringContext<'a> {
    crate_root: Option<Symbol>,

    /// Used to assign ids to HIR nodes that do not directly correspond to an AST node.
    sess: &'a Session,

    cstore: &'a dyn CrateStore,

    resolver: &'a mut dyn Resolver,

    /// The items being lowered are collected here.
    items: BTreeMap<hir::HirId, hir::Item>,

    trait_items: BTreeMap<hir::TraitItemId, hir::TraitItem>,
    impl_items: BTreeMap<hir::ImplItemId, hir::ImplItem>,
    bodies: BTreeMap<hir::BodyId, hir::Body>,
    exported_macros: Vec<hir::MacroDef>,

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

    /// Whether or not in-band lifetimes are being collected. This is used to
    /// indicate whether or not we're in a place where new lifetimes will result
    /// in in-band lifetime definitions, such a function or an impl header,
    /// including implicit lifetimes from `impl_header_lifetime_elision`.
    is_collecting_in_band_lifetimes: bool,

    /// Currently in-scope lifetimes defined in impl headers, fn headers, or HRTB.
    /// When `is_collectin_in_band_lifetimes` is true, each lifetime is checked
    /// against this list to see if it is already in-scope, or if a definition
    /// needs to be created for it.
    in_scope_lifetimes: Vec<Ident>,

    current_module: NodeId,

    type_def_lifetime_params: DefIdMap<usize>,

    current_hir_id_owner: Vec<(DefIndex, u32)>,
    item_local_id_counters: NodeMap<u32>,
    node_id_to_hir_id: IndexVec<NodeId, hir::HirId>,

    allow_try_trait: Option<Lrc<[Symbol]>>,
    allow_gen_future: Option<Lrc<[Symbol]>>,
}

pub trait Resolver {
    /// Resolve a path generated by the lowerer when expanding `for`, `if let`, etc.
    fn resolve_ast_path(
        &mut self,
        path: &ast::Path,
        is_value: bool,
    ) -> Res<NodeId>;

    /// Obtain resolution for a `NodeId` with a single resolution.
    fn get_partial_res(&mut self, id: NodeId) -> Option<PartialRes>;

    /// Obtain per-namespace resolutions for `use` statement with the given `NoedId`.
    fn get_import_res(&mut self, id: NodeId) -> PerNS<Option<Res<NodeId>>>;

    /// Obtain resolution for a label with the given `NodeId`.
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
        is_value: bool,
    ) -> (ast::Path, Res<NodeId>);
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

    /// Treat `impl Trait` as shorthand for a new existential parameter.
    /// Example: `fn foo() -> impl Debug`, where `impl Debug` is conceptually
    /// equivalent to a fresh existential parameter like `existential type T; fn foo() -> T`.
    ///
    /// We optionally store a `DefId` for the parent item here so we can look up necessary
    /// information later. It is `None` when no information about the context should be stored
    /// (e.g., for consts and statics).
    Existential(Option<DefId> /* fn def-ID */),

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
            Existential(fn_def_id) => Existential(*fn_def_id),
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
        crate_root: std_inject::injected_crate_name().map(Symbol::intern),
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
#[derive(Copy, Clone)]
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

    /// Used in the return types of `async fn` where there exists
    /// exactly one argument-position elided lifetime.
    ///
    /// In `async fn`, we lower the arguments types using the `CreateParameter`
    /// mode, meaning that non-`dyn` elided lifetimes are assigned a fresh name.
    /// If any corresponding elided lifetimes appear in the output, we need to
    /// replace them with references to the fresh name assigned to the corresponding
    /// elided lifetime in the arguments.
    ///
    /// For **Modern cases**, replace the anonymous parameter with a
    /// reference to a specific freshly-named lifetime that was
    /// introduced in argument
    ///
    /// For **Dyn Bound** cases, pass responsibility to
    /// `resole_lifetime` code.
    Replace(LtReplacement),
}

/// The type of elided lifetime replacement to perform on `async fn` return types.
#[derive(Copy, Clone)]
enum LtReplacement {
    /// Fresh name introduced by the single non-dyn elided lifetime
    /// in the arguments of the async fn.
    Some(ParamName),

    /// There is no single non-dyn elided lifetime because no lifetimes
    /// appeared in the arguments.
    NoLifetimes,

    /// There is no single non-dyn elided lifetime because multiple
    /// lifetimes appeared in the arguments.
    MultipleLifetimes,
}

/// Calculates the `LtReplacement` to use for elided lifetimes in the return
/// type based on the fresh elided lifetimes introduced in argument position.
fn get_elided_lt_replacement(arg_position_lifetimes: &[(Span, ParamName)]) -> LtReplacement {
    match arg_position_lifetimes {
        [] => LtReplacement::NoLifetimes,
        [(_span, param)] => LtReplacement::Some(*param),
        _ => LtReplacement::MultipleLifetimes,
    }
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
                                Mark::root(),
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
                match p.node {
                    // Doesn't generate a HIR node
                    PatKind::Paren(..) => {},
                    _ => {
                        if let Some(owner) = self.hir_id_owner {
                            self.lctx.lower_node_id_with_owner(p.id, owner);
                        }
                    }
                };

                visit::walk_pat(self, p)
            }

            fn visit_item(&mut self, item: &'tcx Item) {
                let hir_id = self.lctx.allocate_hir_id_counter(item.id);

                match item.node {
                    ItemKind::Struct(_, ref generics)
                    | ItemKind::Union(_, ref generics)
                    | ItemKind::Enum(_, ref generics)
                    | ItemKind::Ty(_, ref generics)
                    | ItemKind::Existential(_, ref generics)
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
                        for argument in &f.decl.inputs {
                            // We don't lower the ids of argument patterns
                            self.with_hir_id_owner(None, |this| {
                                this.visit_pat(&argument.pat);
                            });
                            self.visit_ty(&argument.ty)
                        }
                        self.visit_fn_ret_ty(&f.decl.output)
                    }
                    _ => visit::walk_ty(self, t),
                }
            }
        }

        struct ItemLowerer<'tcx, 'interner> {
            lctx: &'tcx mut LoweringContext<'interner>,
        }

        impl<'tcx, 'interner> ItemLowerer<'tcx, 'interner> {
            fn with_trait_impl_ref<F>(&mut self, trait_impl_ref: &Option<TraitRef>, f: F)
            where
                F: FnOnce(&mut Self),
            {
                let old = self.lctx.is_in_trait_impl;
                self.lctx.is_in_trait_impl = if let &None = trait_impl_ref {
                    false
                } else {
                    true
                };
                f(self);
                self.lctx.is_in_trait_impl = old;
            }
        }

        impl<'tcx, 'interner> Visitor<'tcx> for ItemLowerer<'tcx, 'interner> {
            fn visit_mod(&mut self, m: &'tcx Mod, _s: Span, _attrs: &[Attribute], n: NodeId) {
                self.lctx.modules.insert(n, hir::ModuleItems {
                    items: BTreeSet::new(),
                    trait_items: BTreeSet::new(),
                    impl_items: BTreeSet::new(),
                });

                let old = self.lctx.current_module;
                self.lctx.current_module = n;
                visit::walk_mod(self, m);
                self.lctx.current_module = old;
            }

            fn visit_item(&mut self, item: &'tcx Item) {
                let mut item_hir_id = None;
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    if let Some(hir_item) = lctx.lower_item(item) {
                        item_hir_id = Some(hir_item.hir_id);
                        lctx.insert_item(hir_item);
                    }
                });

                if let Some(hir_id) = item_hir_id {
                    self.lctx.with_parent_item_lifetime_defs(hir_id, |this| {
                        let this = &mut ItemLowerer { lctx: this };
                        if let ItemKind::Impl(.., ref opt_trait_ref, _, _) = item.node {
                            this.with_trait_impl_ref(opt_trait_ref, |this| {
                                visit::walk_item(this, item)
                            });
                        } else {
                            visit::walk_item(this, item);
                        }
                    });
                }
            }

            fn visit_trait_item(&mut self, item: &'tcx TraitItem) {
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    let hir_item = lctx.lower_trait_item(item);
                    let id = hir::TraitItemId { hir_id: hir_item.hir_id };
                    lctx.trait_items.insert(id, hir_item);
                    lctx.modules.get_mut(&lctx.current_module).unwrap().trait_items.insert(id);
                });

                visit::walk_trait_item(self, item);
            }

            fn visit_impl_item(&mut self, item: &'tcx ImplItem) {
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    let hir_item = lctx.lower_impl_item(item);
                    let id = hir::ImplItemId { hir_id: hir_item.hir_id };
                    lctx.impl_items.insert(id, hir_item);
                    lctx.modules.get_mut(&lctx.current_module).unwrap().impl_items.insert(id);
                });
                visit::walk_impl_item(self, item);
            }
        }

        self.lower_node_id(CRATE_NODE_ID);
        debug_assert!(self.node_id_to_hir_id[CRATE_NODE_ID] == hir::CRATE_HIR_ID);

        visit::walk_crate(&mut MiscCollector { lctx: &mut self, hir_id_owner: None }, c);
        visit::walk_crate(&mut ItemLowerer { lctx: &mut self }, c);

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

    fn generator_movability_for_fn(
        &mut self,
        decl: &ast::FnDecl,
        fn_decl_span: Span,
        generator_kind: Option<hir::GeneratorKind>,
        movability: Movability,
    ) -> Option<hir::GeneratorMovability> {
        match generator_kind {
            Some(hir::GeneratorKind::Gen) =>  {
                if !decl.inputs.is_empty() {
                    span_err!(
                        self.sess,
                        fn_decl_span,
                        E0628,
                        "generators cannot have explicit arguments"
                    );
                    self.sess.abort_if_errors();
                }
                Some(match movability {
                    Movability::Movable => hir::GeneratorMovability::Movable,
                    Movability::Static => hir::GeneratorMovability::Static,
                })
            },
            Some(hir::GeneratorKind::Async) => {
                bug!("non-`async` closure body turned `async` during lowering");
            },
            None => {
                if movability == Movability::Static {
                    span_err!(
                        self.sess,
                        fn_decl_span,
                        E0697,
                        "closures cannot be static"
                    );
                }
                None
            },
        }
    }

    fn record_body(&mut self, arguments: HirVec<hir::Arg>, value: hir::Expr) -> hir::BodyId {
        let body = hir::Body {
            generator_kind: self.generator_kind,
            arguments,
            value,
        };
        let id = body.id();
        self.bodies.insert(id, body);
        id
    }

    fn next_id(&mut self) -> hir::HirId {
        self.lower_node_id(self.sess.next_node_id())
    }

    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        res.map_id(|id| {
            self.lower_node_id_generic(id, |_| {
                panic!("expected node_id to be lowered already for res {:#?}", res)
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
        reason: CompilerDesugaringKind,
        span: Span,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
    ) -> Span {
        let mark = Mark::fresh(Mark::root());
        mark.set_expn_info(ExpnInfo {
            def_site: Some(span),
            allow_internal_unstable,
            ..ExpnInfo::default(source_map::CompilerDesugaring(reason), span, self.sess.edition())
        });
        span.with_ctxt(SyntaxContext::empty().apply_mark(mark))
    }

    fn with_anonymous_lifetime_mode<R>(
        &mut self,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        op: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old_anonymous_lifetime_mode = self.anonymous_lifetime_mode;
        self.anonymous_lifetime_mode = anonymous_lifetime_mode;
        let result = op(self);
        self.anonymous_lifetime_mode = old_anonymous_lifetime_mode;
        result
    }

    /// Creates a new `hir::GenericParam` for every new lifetime and
    /// type parameter encountered while evaluating `f`. Definitions
    /// are created with the parent provided. If no `parent_id` is
    /// provided, no definitions will be returned.
    ///
    /// Presuming that in-band lifetimes are enabled, then
    /// `self.anonymous_lifetime_mode` will be updated to match the
    /// argument while `f` is running (and restored afterwards).
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
            Mark::root(),
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

        if self.in_scope_lifetimes.contains(&ident.modern()) {
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
            GenericParamKind::Lifetime { .. } => Some(param.ident.modern()),
            _ => None,
        });
        self.in_scope_lifetimes.extend(lt_def_names);

        let res = f(self);

        self.in_scope_lifetimes.truncate(old_len);
        res
    }

    // Same as the method above, but accepts `hir::GenericParam`s
    // instead of `ast::GenericParam`s.
    // This should only be used with generics that have already had their
    // in-band lifetimes added. In practice, this means that this function is
    // only used when lowering a child item of a trait or impl.
    fn with_parent_item_lifetime_defs<T, F>(&mut self,
        parent_hir_id: hir::HirId,
        f: F
    ) -> T where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let old_len = self.in_scope_lifetimes.len();

        let parent_generics = match self.items.get(&parent_hir_id).unwrap().node {
            hir::ItemKind::Impl(_, _, _, ref generics, ..)
            | hir::ItemKind::Trait(_, _, ref generics, ..) => {
                &generics.params[..]
            }
            _ => &[],
        };
        let lt_def_names = parent_generics.iter().filter_map(|param| match param.kind {
            hir::GenericParamKind::Lifetime { .. } => Some(param.name.ident().modern()),
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
    /// argument while `f` is running (and restored afterwards).
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

    fn with_catch_scope<T, F>(&mut self, catch_id: NodeId, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let len = self.catch_scopes.len();
        self.catch_scopes.push(catch_id);

        let result = f(self);
        assert_eq!(
            len + 1,
            self.catch_scopes.len(),
            "catch scopes should be added and removed in stack order"
        );

        self.catch_scopes.pop().unwrap();

        result
    }

    fn make_async_expr(
        &mut self,
        capture_clause: CaptureBy,
        closure_node_id: NodeId,
        ret_ty: Option<syntax::ptr::P<Ty>>,
        span: Span,
        body: impl FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    ) -> hir::ExprKind {
        let capture_clause = self.lower_capture_clause(capture_clause);
        let output = match ret_ty {
            Some(ty) => FunctionRetTy::Ty(ty),
            None => FunctionRetTy::Default(span),
        };
        let ast_decl = FnDecl {
            inputs: vec![],
            output,
            c_variadic: false
        };
        let decl = self.lower_fn_decl(&ast_decl, None, /* impl trait allowed */ false, None);
        let body_id = self.lower_fn_body(&ast_decl, |this| {
            this.generator_kind = Some(hir::GeneratorKind::Async);
            body(this)
        });
        let generator = hir::Expr {
            hir_id: self.lower_node_id(closure_node_id),
            node: hir::ExprKind::Closure(capture_clause, decl, body_id, span,
                Some(hir::GeneratorMovability::Static)),
            span,
            attrs: ThinVec::new(),
        };

        let unstable_span = self.mark_span_with_reason(
            CompilerDesugaringKind::Async,
            span,
            self.allow_gen_future.clone(),
        );
        let gen_future = self.expr_std_path(
            unstable_span, &[sym::future, sym::from_generator], None, ThinVec::new());
        hir::ExprKind::Call(P(gen_future), hir_vec![generator])
    }

    fn lower_body(
        &mut self,
        f: impl FnOnce(&mut LoweringContext<'_>) -> (HirVec<hir::Arg>, hir::Expr),
    ) -> hir::BodyId {
        let prev_gen_kind = self.generator_kind.take();
        let (arguments, result) = f(self);
        let body_id = self.record_body(arguments, result);
        self.generator_kind = prev_gen_kind;
        body_id
    }

    fn lower_fn_body(
        &mut self,
        decl: &FnDecl,
        body: impl FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    ) -> hir::BodyId {
        self.lower_body(|this| (
            decl.inputs.iter().map(|x| this.lower_arg(x)).collect(),
            body(this),
        ))
    }

    fn lower_const_body(&mut self, expr: &Expr) -> hir::BodyId {
        self.lower_body(|this| (hir_vec![], this.lower_expr(expr)))
    }

    fn with_loop_scope<T, F>(&mut self, loop_id: NodeId, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        // We're no longer in the base loop's condition; we're in another loop.
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let len = self.loop_scopes.len();
        self.loop_scopes.push(loop_id);

        let result = f(self);
        assert_eq!(
            len + 1,
            self.loop_scopes.len(),
            "loop scopes should be added and removed in stack order"
        );

        self.loop_scopes.pop().unwrap();

        self.is_in_loop_condition = was_in_loop_condition;

        result
    }

    fn with_loop_condition_scope<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = true;

        let result = f(self);

        self.is_in_loop_condition = was_in_loop_condition;

        result
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

    fn lower_label(&mut self, label: Option<Label>) -> Option<hir::Label> {
        label.map(|label| hir::Label {
            ident: label.ident,
        })
    }

    fn lower_loop_destination(&mut self, destination: Option<(NodeId, Label)>) -> hir::Destination {
        let target_id = match destination {
            Some((id, _)) => {
                if let Some(loop_id) = self.resolver.get_label_res(id) {
                    Ok(self.lower_node_id(loop_id))
                } else {
                    Err(hir::LoopIdError::UnresolvedLabel)
                }
            }
            None => {
                self.loop_scopes
                    .last()
                    .cloned()
                    .map(|id| Ok(self.lower_node_id(id)))
                    .unwrap_or(Err(hir::LoopIdError::OutsideLoopScope))
                    .into()
            }
        };
        hir::Destination {
            label: self.lower_label(destination.map(|(_, label)| label)),
            target_id,
        }
    }

    fn lower_attrs(&mut self, attrs: &[Attribute]) -> hir::HirVec<Attribute> {
        attrs
            .iter()
            .map(|a| self.lower_attr(a))
            .collect()
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

    fn lower_arm(&mut self, arm: &Arm) -> hir::Arm {
        hir::Arm {
            hir_id: self.next_id(),
            attrs: self.lower_attrs(&arm.attrs),
            pats: arm.pats.iter().map(|x| self.lower_pat(x)).collect(),
            guard: match arm.guard {
                Some(ref x) => Some(hir::Guard::If(P(self.lower_expr(x)))),
                _ => None,
            },
            body: P(self.lower_expr(&arm.body)),
            span: arm.span,
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
    fn lower_assoc_ty_constraint(&mut self,
                                 c: &AssocTyConstraint,
                                 itctx: ImplTraitContext<'_>)
                                 -> hir::TypeBinding {
        debug!("lower_assoc_ty_constraint(constraint={:?}, itctx={:?})", c, itctx);

        let kind = match c.kind {
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
                    ImplTraitContext::Existential(_) => (true, itctx),

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
                    // then to an existential type).
                    //
                    // FIXME: this is only needed until `impl Trait` is allowed in type aliases.
                    ImplTraitContext::Disallowed(_) if self.is_in_dyn_type =>
                        (true, ImplTraitContext::Existential(None)),

                    // We are in the argument position, but not within a dyn type:
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
                        Mark::root(),
                        DUMMY_SP
                    );

                    self.with_dyn_type_scope(false, |this| {
                        let ty = this.lower_ty(
                            &Ty {
                                id: this.sess.next_node_id(),
                                node: TyKind::ImplTrait(impl_trait_node_id, bounds.clone()),
                                span: DUMMY_SP,
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
            hir_id: self.lower_node_id(c.id),
            ident: c.ident,
            kind,
            span: c.span,
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
                                arg_names: this.lower_fn_args_to_names(&f.decl),
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
                            Ident::with_empty_ctxt(kw::SelfUpper)
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
                    ImplTraitContext::Existential(fn_def_id) => {
                        self.lower_existential_impl_trait(
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
                        let ident = Ident::from_str(&pprust::ty_to_string(t)).with_span_pos(span);
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
                                  "add #![feature(impl_trait_in_bindings)] to the crate attributes \
                                   to enable");
                        }
                        err.emit();
                        hir::TyKind::Err
                    }
                }
            }
            TyKind::Mac(_) => bug!("`TyMac` should have been expanded by now."),
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

    fn lower_existential_impl_trait(
        &mut self,
        span: Span,
        fn_def_id: Option<DefId>,
        exist_ty_node_id: NodeId,
        lower_bounds: impl FnOnce(&mut LoweringContext<'_>) -> hir::GenericBounds,
    ) -> hir::TyKind {
        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let exist_ty_span = self.mark_span_with_reason(
            CompilerDesugaringKind::ExistentialType,
            span,
            None,
        );

        let exist_ty_def_index = self
            .resolver
            .definitions()
            .opt_def_index(exist_ty_node_id)
            .unwrap();

        self.allocate_hir_id_counter(exist_ty_node_id);

        let hir_bounds = self.with_hir_id_owner(exist_ty_node_id, lower_bounds);

        let (lifetimes, lifetime_defs) = self.lifetimes_from_impl_trait_bounds(
            exist_ty_node_id,
            exist_ty_def_index,
            &hir_bounds,
        );

        self.with_hir_id_owner(exist_ty_node_id, |lctx| {
            let exist_ty_item = hir::ExistTy {
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
                origin: hir::ExistTyOrigin::ReturnImplTrait,
            };

            trace!("exist ty from impl trait def-index: {:#?}", exist_ty_def_index);
            let exist_ty_id = lctx.generate_existential_type(
                exist_ty_node_id,
                exist_ty_item,
                span,
                exist_ty_span,
            );

            // `impl Trait` now just becomes `Foo<'a, 'b, ..>`.
            hir::TyKind::Def(hir::ItemId { id: exist_ty_id }, lifetimes)
        })
    }

    /// Registers a new existential type with the proper `NodeId`s and
    /// returns the lowered node-ID for the existential type.
    fn generate_existential_type(
        &mut self,
        exist_ty_node_id: NodeId,
        exist_ty_item: hir::ExistTy,
        span: Span,
        exist_ty_span: Span,
    ) -> hir::HirId {
        let exist_ty_item_kind = hir::ItemKind::Existential(exist_ty_item);
        let exist_ty_id = self.lower_node_id(exist_ty_node_id);
        // Generate an `existential type Foo: Trait;` declaration.
        trace!("registering existential type with id {:#?}", exist_ty_id);
        let exist_ty_item = hir::Item {
            hir_id: exist_ty_id,
            ident: Ident::invalid(),
            attrs: Default::default(),
            node: exist_ty_item_kind,
            vis: respan(span.shrink_to_lo(), hir::VisibilityKind::Inherited),
            span: exist_ty_span,
        };

        // Insert the item into the global item list. This usually happens
        // automatically for all AST items. But this existential type item
        // does not actually exist in the AST.
        self.insert_item(exist_ty_item);
        exist_ty_id
    }

    fn lifetimes_from_impl_trait_bounds(
        &mut self,
        exist_ty_id: NodeId,
        parent_index: DefIndex,
        bounds: &hir::GenericBounds,
    ) -> (HirVec<hir::GenericArg>, HirVec<hir::GenericParam>) {
        // This visitor walks over `impl Trait` bounds and creates defs for all lifetimes that
        // appear in the bounds, excluding lifetimes that are created within the bounds.
        // E.g., `'a`, `'b`, but not `'c` in `impl for<'c> SomeTrait<'a, 'b, 'c>`.
        struct ImplTraitLifetimeCollector<'r, 'a> {
            context: &'r mut LoweringContext<'a>,
            parent: DefIndex,
            exist_ty_id: NodeId,
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
                            // `abstract type Foo<'_>: SomeTrait<'_>;`.
                            hir::LifetimeName::Underscore
                        } else {
                            return;
                        }
                    }
                    hir::LifetimeName::Param(_) => lifetime.name,
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
                        self.context.lower_node_id_with_owner(def_node_id, self.exist_ty_id);
                    self.context.resolver.definitions().create_def_with_parent(
                        self.parent,
                        def_node_id,
                        DefPathData::LifetimeNs(name.ident().as_interned_str()),
                        Mark::root(),
                        lifetime.span);

                    let (name, kind) = match name {
                        hir::LifetimeName::Underscore => (
                            hir::ParamName::Plain(Ident::with_empty_ctxt(kw::UnderscoreLifetime)),
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
            exist_ty_id,
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

    fn lower_foreign_mod(&mut self, fm: &ForeignMod) -> hir::ForeignMod {
        hir::ForeignMod {
            abi: fm.abi,
            items: fm.items
                .iter()
                .map(|x| self.lower_foreign_item(x))
                .collect(),
        }
    }

    fn lower_global_asm(&mut self, ga: &GlobalAsm) -> P<hir::GlobalAsm> {
        P(hir::GlobalAsm {
            asm: ga.asm,
            ctxt: ga.ctxt,
        })
    }

    fn lower_variant(&mut self, v: &Variant) -> hir::Variant {
        Spanned {
            node: hir::VariantKind {
                ident: v.node.ident,
                id: self.lower_node_id(v.node.id),
                attrs: self.lower_attrs(&v.node.attrs),
                data: self.lower_variant_data(&v.node.data),
                disr_expr: v.node.disr_expr.as_ref().map(|e| self.lower_anon_const(e)),
            },
            span: v.span,
        }
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
                                err.span_suggestion(
                                    data.span,
                                    "use angle brackets instead",
                                    format!("<{}>", &snippet[1..snippet.len() - 1]),
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
                    AnonymousLifetimeMode::ReportError |
                    AnonymousLifetimeMode::Replace(_) => {
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
                                ident: Ident::with_empty_ctxt(FN_OUTPUT_NAME),
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
                        ImplTraitContext::Existential(Some(parent_def_id))
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

    fn lower_arg(&mut self, arg: &Arg) -> hir::Arg {
        hir::Arg {
            hir_id: self.lower_node_id(arg.id),
            pat: self.lower_pat(&arg.pat),
        }
    }

    fn lower_fn_args_to_names(&mut self, decl: &FnDecl) -> hir::HirVec<Ident> {
        decl.inputs
            .iter()
            .map(|arg| match arg.pat.node {
                PatKind::Ident(_, ident, _) => ident,
                _ => Ident::new(kw::Invalid, arg.pat.span),
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
            // they can be applied to the existential return type.
            AnonymousLifetimeMode::CreateParameter
        } else {
            self.anonymous_lifetime_mode
        };

        // Remember how many lifetimes were already around so that we can
        // only look at the lifetime parameters introduced by the arguments.
        let lifetime_count_before_args = self.lifetimes_to_define.len();
        let inputs = self.with_anonymous_lifetime_mode(lt_mode, |this| {
            decl.inputs
                .iter()
                .map(|arg| {
                    if let Some((_, ibty)) = &mut in_band_ty_params {
                        this.lower_ty_direct(&arg.ty, ImplTraitContext::Universal(ibty))
                    } else {
                        this.lower_ty_direct(&arg.ty, ImplTraitContext::disallowed())
                    }
                })
                .collect::<HirVec<_>>()
        });

        let output = if let Some(ret_id) = make_ret_async {
            // Calculate the `LtReplacement` to use for any return-position elided
            // lifetimes based on the elided lifetime parameters introduced in the args.
            let lt_replacement = get_elided_lt_replacement(
                &self.lifetimes_to_define[lifetime_count_before_args..]
            );
            self.lower_async_fn_ret_ty(
                &decl.output,
                in_band_ty_params.expect("`make_ret_async` but no `fn_def_id`").0,
                ret_id,
                lt_replacement,
            )
        } else {
            match decl.output {
                FunctionRetTy::Ty(ref ty) => match in_band_ty_params {
                    Some((def_id, _)) if impl_trait_return_allow => {
                        hir::Return(self.lower_ty(ty,
                            ImplTraitContext::Existential(Some(def_id))
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

    // Transforms `-> T` for `async fn` into `-> ExistTy { .. }`
    // combined with the following definition of `ExistTy`:
    //
    //     existential type ExistTy<generics_from_parent_fn>: Future<Output = T>;
    //
    // `inputs`: lowered types of arguments to the function (used to collect lifetimes)
    // `output`: unlowered output type (`T` in `-> T`)
    // `fn_def_id`: `DefId` of the parent function (used to create child impl trait definition)
    // `exist_ty_node_id`: `NodeId` of the existential type that should be created
    // `elided_lt_replacement`: replacement for elided lifetimes in the return type
    fn lower_async_fn_ret_ty(
        &mut self,
        output: &FunctionRetTy,
        fn_def_id: DefId,
        exist_ty_node_id: NodeId,
        elided_lt_replacement: LtReplacement,
    ) -> hir::FunctionRetTy {
        let span = output.span();

        let exist_ty_span = self.mark_span_with_reason(
            CompilerDesugaringKind::Async,
            span,
            None,
        );

        let exist_ty_def_index = self
            .resolver
            .definitions()
            .opt_def_index(exist_ty_node_id)
            .unwrap();

        self.allocate_hir_id_counter(exist_ty_node_id);

        let (exist_ty_id, lifetime_params) = self.with_hir_id_owner(exist_ty_node_id, |this| {
            let future_bound = this.with_anonymous_lifetime_mode(
                AnonymousLifetimeMode::Replace(elided_lt_replacement),
                |this| this.lower_async_fn_output_type_to_future_bound(
                    output,
                    fn_def_id,
                    span,
                ),
            );

            // Calculate all the lifetimes that should be captured
            // by the existential type. This should include all in-scope
            // lifetime parameters, including those defined in-band.
            //
            // Note: this must be done after lowering the output type,
            // as the output type may introduce new in-band lifetimes.
            let lifetime_params: Vec<(Span, ParamName)> =
                this.in_scope_lifetimes
                    .iter().cloned()
                    .map(|ident| (ident.span, ParamName::Plain(ident)))
                    .chain(this.lifetimes_to_define.iter().cloned())
                    .collect();

            let generic_params =
                lifetime_params
                    .iter().cloned()
                    .map(|(span, hir_name)| {
                        this.lifetime_to_generic_param(span, hir_name, exist_ty_def_index)
                    })
                    .collect();

            let exist_ty_item = hir::ExistTy {
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
                origin: hir::ExistTyOrigin::AsyncFn,
            };

            trace!("exist ty from async fn def index: {:#?}", exist_ty_def_index);
            let exist_ty_id = this.generate_existential_type(
                exist_ty_node_id,
                exist_ty_item,
                span,
                exist_ty_span,
            );

            (exist_ty_id, lifetime_params)
        });

        let generic_args =
            lifetime_params
                .iter().cloned()
                .map(|(span, hir_name)| {
                    GenericArg::Lifetime(hir::Lifetime {
                        hir_id: self.next_id(),
                        span,
                        name: hir::LifetimeName::Param(hir_name),
                    })
                })
                .collect();

        let exist_ty_ref = hir::TyKind::Def(hir::ItemId { id: exist_ty_id }, generic_args);

        hir::FunctionRetTy::Return(P(hir::Ty {
            node: exist_ty_ref,
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
                self.lower_ty(ty, ImplTraitContext::Existential(Some(fn_def_id)))
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
                ident: Ident::with_empty_ctxt(FN_OUTPUT_NAME),
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

                    AnonymousLifetimeMode::Replace(replacement) => {
                        let hir_id = self.lower_node_id(l.id);
                        self.replace_elided_lifetime(hir_id, span, replacement)
                    }
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

    /// Replace a return-position elided lifetime with the elided lifetime
    /// from the arguments.
    fn replace_elided_lifetime(
        &mut self,
        hir_id: hir::HirId,
        span: Span,
        replacement: LtReplacement,
    ) -> hir::Lifetime {
        let multiple_or_none = match replacement {
            LtReplacement::Some(name) => {
                return hir::Lifetime {
                    hir_id,
                    span,
                    name: hir::LifetimeName::Param(name),
                };
            }
            LtReplacement::MultipleLifetimes => "multiple",
            LtReplacement::NoLifetimes => "none",
        };

        let mut err = crate::middle::resolve_lifetime::report_missing_lifetime_specifiers(
            self.sess,
            span,
            1,
        );
        err.note(&format!(
            "return-position elided lifetimes require exactly one \
             input-position elided lifetime, found {}.", multiple_or_none));
        err.emit();

        hir::Lifetime { hir_id, span, name: hir::LifetimeName::Error }
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
                    hir::LifetimeName::Error => ParamName::Error,
                };

                let kind = hir::GenericParamKind::Lifetime {
                    kind: hir::LifetimeParamKind::Explicit
                };

                self.is_collecting_in_band_lifetimes = was_collecting_in_band;

                (param_name, kind)
            }
            GenericParamKind::Type { ref default, .. } => {
                // Don't expose `Self` (recovered "keyword used as ident" parse error).
                // `rustc::ty` expects `Self` to be only used for a trait's `Self`.
                // Instead, use `gensym("Self")` to create a distinct name that looks the same.
                let ident = if param.ident.name == kw::SelfUpper {
                    param.ident.gensym()
                } else {
                    param.ident
                };

                let add_bounds = add_bounds.get(&param.id).map_or(&[][..], |x| &x);
                if !add_bounds.is_empty() {
                    let params = self.lower_param_bounds(add_bounds, itctx.reborrow()).into_iter();
                    bounds = bounds.into_iter()
                                   .chain(params)
                                   .collect();
                }

                let kind = hir::GenericParamKind::Type {
                    default: default.as_ref().map(|x| {
                        self.lower_ty(x, ImplTraitContext::Existential(None))
                    }),
                    synthetic: param.attrs.iter()
                                          .filter(|attr| attr.check_name(sym::rustc_synthetic))
                                          .map(|_| hir::SyntheticTyParamKind::ImplTrait)
                                          .next(),
                };

                (hir::ParamName::Plain(ident), kind)
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

    fn lower_generics(
        &mut self,
        generics: &Generics,
        itctx: ImplTraitContext<'_>)
        -> hir::Generics
    {
        // Collect `?Trait` bounds in where clause and move them to parameter definitions.
        // FIXME: this could probably be done with less rightward drift. It also looks like two
        // control paths where `report_error` is called are the only paths that advance to after the
        // match statement, so the error reporting could probably just be moved there.
        let mut add_bounds: NodeMap<Vec<_>> = Default::default();
        for pred in &generics.where_clause.predicates {
            if let WherePredicate::BoundPredicate(ref bound_pred) = *pred {
                'next_bound: for bound in &bound_pred.bounds {
                    if let GenericBound::Trait(_, TraitBoundModifier::Maybe) = *bound {
                        let report_error = |this: &mut Self| {
                            this.diagnostic().span_err(
                                bound_pred.bounded_ty.span,
                                "`?Trait` bounds are only permitted at the \
                                 point where a type parameter is declared",
                            );
                        };
                        // Check if the where clause type is a plain type parameter.
                        match bound_pred.bounded_ty.node {
                            TyKind::Path(None, ref path)
                                if path.segments.len() == 1
                                    && bound_pred.bound_generic_params.is_empty() =>
                            {
                                if let Some(Res::Def(DefKind::TyParam, def_id)) = self.resolver
                                    .get_partial_res(bound_pred.bounded_ty.id)
                                    .map(|d| d.base_res())
                                {
                                    if let Some(node_id) =
                                        self.resolver.definitions().as_local_node_id(def_id)
                                    {
                                        for param in &generics.params {
                                            match param.kind {
                                                GenericParamKind::Type { .. } => {
                                                    if node_id == param.id {
                                                        add_bounds.entry(param.id)
                                                            .or_default()
                                                            .push(bound.clone());
                                                        continue 'next_bound;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                report_error(self)
                            }
                            _ => report_error(self),
                        }
                    }
                }
            }
        }

        hir::Generics {
            params: self.lower_generic_params(&generics.params, &add_bounds, itctx),
            where_clause: self.lower_where_clause(&generics.where_clause),
            span: generics.span,
        }
    }

    fn lower_where_clause(&mut self, wc: &WhereClause) -> hir::WhereClause {
        self.with_anonymous_lifetime_mode(
            AnonymousLifetimeMode::ReportError,
            |this| {
                hir::WhereClause {
                    predicates: wc.predicates
                        .iter()
                        .map(|predicate| this.lower_where_predicate(predicate))
                        .collect(),
                    span: wc.span,
                }
            },
        )
    }

    fn lower_where_predicate(&mut self, pred: &WherePredicate) -> hir::WherePredicate {
        match *pred {
            WherePredicate::BoundPredicate(WhereBoundPredicate {
                ref bound_generic_params,
                ref bounded_ty,
                ref bounds,
                span,
            }) => {
                self.with_in_scope_lifetime_defs(
                    &bound_generic_params,
                    |this| {
                        hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                            bound_generic_params: this.lower_generic_params(
                                bound_generic_params,
                                &NodeMap::default(),
                                ImplTraitContext::disallowed(),
                            ),
                            bounded_ty: this.lower_ty(bounded_ty, ImplTraitContext::disallowed()),
                            bounds: bounds
                                .iter()
                                .filter_map(|bound| match *bound {
                                    // Ignore `?Trait` bounds.
                                    // They were copied into type parameters already.
                                    GenericBound::Trait(_, TraitBoundModifier::Maybe) => None,
                                    _ => Some(this.lower_param_bound(
                                        bound,
                                        ImplTraitContext::disallowed(),
                                    )),
                                })
                                .collect(),
                            span,
                        })
                    },
                )
            }
            WherePredicate::RegionPredicate(WhereRegionPredicate {
                ref lifetime,
                ref bounds,
                span,
            }) => hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span,
                lifetime: self.lower_lifetime(lifetime),
                bounds: self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
            }),
            WherePredicate::EqPredicate(WhereEqPredicate {
                id,
                ref lhs_ty,
                ref rhs_ty,
                span,
            }) => {
                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    hir_id: self.lower_node_id(id),
                    lhs_ty: self.lower_ty(lhs_ty, ImplTraitContext::disallowed()),
                    rhs_ty: self.lower_ty(rhs_ty, ImplTraitContext::disallowed()),
                    span,
                })
            },
        }
    }

    fn lower_variant_data(&mut self, vdata: &VariantData) -> hir::VariantData {
        match *vdata {
            VariantData::Struct(ref fields, recovered) => hir::VariantData::Struct(
                fields.iter().enumerate().map(|f| self.lower_struct_field(f)).collect(),
                recovered,
            ),
            VariantData::Tuple(ref fields, id) => {
                hir::VariantData::Tuple(
                    fields
                        .iter()
                        .enumerate()
                        .map(|f| self.lower_struct_field(f))
                        .collect(),
                    self.lower_node_id(id),
                )
            },
            VariantData::Unit(id) => {
                hir::VariantData::Unit(self.lower_node_id(id))
            },
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

    fn lower_struct_field(&mut self, (index, f): (usize, &StructField)) -> hir::StructField {
        let ty = if let TyKind::Path(ref qself, ref path) = f.ty.node {
            let t = self.lower_path_ty(
                &f.ty,
                qself,
                path,
                ParamMode::ExplicitNamed, // no `'_` in declarations (Issue #61124)
                ImplTraitContext::disallowed()
            );
            P(t)
        } else {
            self.lower_ty(&f.ty, ImplTraitContext::disallowed())
        };
        hir::StructField {
            span: f.span,
            hir_id: self.lower_node_id(f.id),
            ident: match f.ident {
                Some(ident) => ident,
                // FIXME(jseyfried): positional field hygiene.
                None => Ident::new(sym::integer(index), f.span),
            },
            vis: self.lower_visibility(&f.vis, None),
            ty,
            attrs: self.lower_attrs(&f.attrs),
        }
    }

    fn lower_field(&mut self, f: &Field) -> hir::Field {
        hir::Field {
            hir_id: self.next_id(),
            ident: f.ident,
            expr: P(self.lower_expr(&f.expr)),
            span: f.span,
            is_shorthand: f.is_shorthand,
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

    fn lower_block_with_stmts(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
        mut stmts: Vec<hir::Stmt>,
    ) -> P<hir::Block> {
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

    fn lower_block(&mut self, b: &Block, targeted_by_break: bool) -> P<hir::Block> {
        self.lower_block_with_stmts(b, targeted_by_break, vec![])
    }

    fn lower_maybe_async_body(
        &mut self,
        decl: &FnDecl,
        asyncness: IsAsync,
        body: &Block,
    ) -> hir::BodyId {
        let closure_id = match asyncness {
            IsAsync::Async { closure_id, .. } => closure_id,
            IsAsync::NotAsync => return self.lower_fn_body(&decl, |this| {
                let body = this.lower_block(body, false);
                this.expr_block(body, ThinVec::new())
            }),
        };

        self.lower_body(|this| {
            let mut arguments: Vec<hir::Arg> = Vec::new();
            let mut statements: Vec<hir::Stmt> = Vec::new();

            // Async function arguments are lowered into the closure body so that they are
            // captured and so that the drop order matches the equivalent non-async functions.
            //
            // from:
            //
            //     async fn foo(<pattern>: <ty>, <pattern>: <ty>, <pattern>: <ty>) {
            //       async move {
            //       }
            //     }
            //
            // into:
            //
            //     fn foo(__arg0: <ty>, __arg1: <ty>, __arg2: <ty>) {
            //       async move {
            //         let __arg2 = __arg2;
            //         let <pattern> = __arg2;
            //         let __arg1 = __arg1;
            //         let <pattern> = __arg1;
            //         let __arg0 = __arg0;
            //         let <pattern> = __arg0;
            //       }
            //     }
            //
            // If `<pattern>` is a simple ident, then it is lowered to a single
            // `let <pattern> = <pattern>;` statement as an optimization.
            for (index, argument) in decl.inputs.iter().enumerate() {
                let argument = this.lower_arg(argument);
                let span = argument.pat.span;

                // Check if this is a binding pattern, if so, we can optimize and avoid adding a
                // `let <pat> = __argN;` statement. In this case, we do not rename the argument.
                let (ident, is_simple_argument) = match argument.pat.node {
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, ident, _) =>
                        (ident, true),
                    _ => {
                        // Replace the ident for bindings that aren't simple.
                        let name = format!("__arg{}", index);
                        let ident = Ident::from_str(&name);

                        (ident, false)
                    },
                };

                let desugared_span =
                    this.mark_span_with_reason(CompilerDesugaringKind::Async, span, None);

                // Construct an argument representing `__argN: <ty>` to replace the argument of the
                // async function.
                //
                // If this is the simple case, this argument will end up being the same as the
                // original argument, but with a different pattern id.
                let (new_argument_pat, new_argument_id) = this.pat_ident(desugared_span, ident);
                let new_argument = hir::Arg {
                    hir_id: argument.hir_id,
                    pat: new_argument_pat,
                };

                if is_simple_argument {
                    // If this is the simple case, then we only insert one statement that is
                    // `let <pat> = <pat>;`. We re-use the original argument's pattern so that
                    // `HirId`s are densely assigned.
                    let expr = this.expr_ident(desugared_span, ident, new_argument_id);
                    let stmt = this.stmt_let_pat(
                        desugared_span, Some(P(expr)), argument.pat, hir::LocalSource::AsyncFn);
                    statements.push(stmt);
                } else {
                    // If this is not the simple case, then we construct two statements:
                    //
                    // ```
                    // let __argN = __argN;
                    // let <pat> = __argN;
                    // ```
                    //
                    // The first statement moves the argument into the closure and thus ensures
                    // that the drop order is correct.
                    //
                    // The second statement creates the bindings that the user wrote.

                    // Construct the `let mut __argN = __argN;` statement. It must be a mut binding
                    // because the user may have specified a `ref mut` binding in the next
                    // statement.
                    let (move_pat, move_id) = this.pat_ident_binding_mode(
                        desugared_span, ident, hir::BindingAnnotation::Mutable);
                    let move_expr = this.expr_ident(desugared_span, ident, new_argument_id);
                    let move_stmt = this.stmt_let_pat(
                        desugared_span, Some(P(move_expr)), move_pat, hir::LocalSource::AsyncFn);

                    // Construct the `let <pat> = __argN;` statement. We re-use the original
                    // argument's pattern so that `HirId`s are densely assigned.
                    let pattern_expr = this.expr_ident(desugared_span, ident, move_id);
                    let pattern_stmt = this.stmt_let_pat(
                        desugared_span, Some(P(pattern_expr)), argument.pat,
                        hir::LocalSource::AsyncFn);

                    statements.push(move_stmt);
                    statements.push(pattern_stmt);
                };

                arguments.push(new_argument);
            }

            let async_expr = this.make_async_expr(
                CaptureBy::Value, closure_id, None, body.span,
                |this| {
                    let body = this.lower_block_with_stmts(body, false, statements);
                    this.expr_block(body, ThinVec::new())
                });
            (HirVec::from(arguments), this.expr(body.span, async_expr, ThinVec::new()))
        })
    }

    fn lower_item_kind(
        &mut self,
        id: NodeId,
        ident: &mut Ident,
        attrs: &hir::HirVec<Attribute>,
        vis: &mut hir::Visibility,
        i: &ItemKind,
    ) -> hir::ItemKind {
        match *i {
            ItemKind::ExternCrate(orig_name) => hir::ItemKind::ExternCrate(orig_name),
            ItemKind::Use(ref use_tree) => {
                // Start with an empty prefix.
                let prefix = Path {
                    segments: vec![],
                    span: use_tree.span,
                };

                self.lower_use_tree(use_tree, &prefix, id, vis, ident, attrs)
            }
            ItemKind::Static(ref t, m, ref e) => {
                hir::ItemKind::Static(
                    self.lower_ty(
                        t,
                        if self.sess.features_untracked().impl_trait_in_bindings {
                            ImplTraitContext::Existential(None)
                        } else {
                            ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                        }
                    ),
                    self.lower_mutability(m),
                    self.lower_const_body(e),
                )
            }
            ItemKind::Const(ref t, ref e) => {
                hir::ItemKind::Const(
                    self.lower_ty(
                        t,
                        if self.sess.features_untracked().impl_trait_in_bindings {
                            ImplTraitContext::Existential(None)
                        } else {
                            ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                        }
                    ),
                    self.lower_const_body(e)
                )
            }
            ItemKind::Fn(ref decl, header, ref generics, ref body) => {
                let fn_def_id = self.resolver.definitions().local_def_id(id);
                self.with_new_scopes(|this| {
                    this.current_item = Some(ident.span);

                    // Note: we don't need to change the return type from `T` to
                    // `impl Future<Output = T>` here because lower_body
                    // only cares about the input argument patterns in the function
                    // declaration (decl), not the return types.
                    let body_id = this.lower_maybe_async_body(&decl, header.asyncness.node, body);

                    let (generics, fn_decl) = this.add_in_band_defs(
                        generics,
                        fn_def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, idty| this.lower_fn_decl(
                            &decl,
                            Some((fn_def_id, idty)),
                            true,
                            header.asyncness.node.opt_return_id()
                        ),
                    );

                    hir::ItemKind::Fn(
                        fn_decl,
                        this.lower_fn_header(header),
                        generics,
                        body_id,
                    )
                })
            }
            ItemKind::Mod(ref m) => hir::ItemKind::Mod(self.lower_mod(m)),
            ItemKind::ForeignMod(ref nm) => hir::ItemKind::ForeignMod(self.lower_foreign_mod(nm)),
            ItemKind::GlobalAsm(ref ga) => hir::ItemKind::GlobalAsm(self.lower_global_asm(ga)),
            ItemKind::Ty(ref t, ref generics) => hir::ItemKind::Ty(
                self.lower_ty(t, ImplTraitContext::disallowed()),
                self.lower_generics(generics, ImplTraitContext::disallowed()),
            ),
            ItemKind::Existential(ref b, ref generics) => hir::ItemKind::Existential(
                hir::ExistTy {
                    generics: self.lower_generics(generics,
                        ImplTraitContext::Existential(None)),
                    bounds: self.lower_param_bounds(b,
                        ImplTraitContext::Existential(None)),
                    impl_trait_fn: None,
                    origin: hir::ExistTyOrigin::ExistentialType,
                },
            ),
            ItemKind::Enum(ref enum_definition, ref generics) => {
                hir::ItemKind::Enum(
                    hir::EnumDef {
                        variants: enum_definition
                            .variants
                            .iter()
                            .map(|x| self.lower_variant(x))
                            .collect(),
                    },
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            },
            ItemKind::Struct(ref struct_def, ref generics) => {
                let struct_def = self.lower_variant_data(struct_def);
                hir::ItemKind::Struct(
                    struct_def,
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            }
            ItemKind::Union(ref vdata, ref generics) => {
                let vdata = self.lower_variant_data(vdata);
                hir::ItemKind::Union(
                    vdata,
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            }
            ItemKind::Impl(
                unsafety,
                polarity,
                defaultness,
                ref ast_generics,
                ref trait_ref,
                ref ty,
                ref impl_items,
            ) => {
                let def_id = self.resolver.definitions().local_def_id(id);

                // Lower the "impl header" first. This ordering is important
                // for in-band lifetimes! Consider `'a` here:
                //
                //     impl Foo<'a> for u32 {
                //         fn method(&'a self) { .. }
                //     }
                //
                // Because we start by lowering the `Foo<'a> for u32`
                // part, we will add `'a` to the list of generics on
                // the impl. When we then encounter it later in the
                // method, it will not be considered an in-band
                // lifetime to be added, but rather a reference to a
                // parent lifetime.
                let lowered_trait_impl_id = self.lower_node_id(id);
                let (generics, (trait_ref, lowered_ty)) = self.add_in_band_defs(
                    ast_generics,
                    def_id,
                    AnonymousLifetimeMode::CreateParameter,
                    |this, _| {
                        let trait_ref = trait_ref.as_ref().map(|trait_ref| {
                            this.lower_trait_ref(trait_ref, ImplTraitContext::disallowed())
                        });

                        if let Some(ref trait_ref) = trait_ref {
                            if let Res::Def(DefKind::Trait, def_id) = trait_ref.path.res {
                                this.trait_impls.entry(def_id).or_default().push(
                                    lowered_trait_impl_id);
                            }
                        }

                        let lowered_ty = this.lower_ty(ty, ImplTraitContext::disallowed());

                        (trait_ref, lowered_ty)
                    },
                );

                let new_impl_items = self.with_in_scope_lifetime_defs(
                    &ast_generics.params,
                    |this| {
                        impl_items
                            .iter()
                            .map(|item| this.lower_impl_item_ref(item))
                            .collect()
                    },
                );

                hir::ItemKind::Impl(
                    self.lower_unsafety(unsafety),
                    self.lower_impl_polarity(polarity),
                    self.lower_defaultness(defaultness, true /* [1] */),
                    generics,
                    trait_ref,
                    lowered_ty,
                    new_impl_items,
                )
            }
            ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref items) => {
                let bounds = self.lower_param_bounds(bounds, ImplTraitContext::disallowed());
                let items = items
                    .iter()
                    .map(|item| self.lower_trait_item_ref(item))
                    .collect();
                hir::ItemKind::Trait(
                    self.lower_is_auto(is_auto),
                    self.lower_unsafety(unsafety),
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                    bounds,
                    items,
                )
            }
            ItemKind::TraitAlias(ref generics, ref bounds) => hir::ItemKind::TraitAlias(
                self.lower_generics(generics, ImplTraitContext::disallowed()),
                self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
            ),
            ItemKind::MacroDef(..)
            | ItemKind::Mac(..) => bug!("`TyMac` should have been expanded by now"),
        }

        // [1] `defaultness.has_value()` is never called for an `impl`, always `true` in order to
        //     not cause an assertion failure inside the `lower_defaultness` function.
    }

    fn lower_use_tree(
        &mut self,
        tree: &UseTree,
        prefix: &Path,
        id: NodeId,
        vis: &mut hir::Visibility,
        ident: &mut Ident,
        attrs: &hir::HirVec<Attribute>,
    ) -> hir::ItemKind {
        debug!("lower_use_tree(tree={:?})", tree);
        debug!("lower_use_tree: vis = {:?}", vis);

        let path = &tree.prefix;
        let segments = prefix
            .segments
            .iter()
            .chain(path.segments.iter())
            .cloned()
            .collect();

        match tree.kind {
            UseTreeKind::Simple(rename, id1, id2) => {
                *ident = tree.ident();

                // First, apply the prefix to the path.
                let mut path = Path {
                    segments,
                    span: path.span,
                };

                // Correctly resolve `self` imports.
                if path.segments.len() > 1
                    && path.segments.last().unwrap().ident.name == kw::SelfLower
                {
                    let _ = path.segments.pop();
                    if rename.is_none() {
                        *ident = path.segments.last().unwrap().ident;
                    }
                }

                let mut resolutions = self.expect_full_res_from_use(id);
                // We want to return *something* from this function, so hold onto the first item
                // for later.
                let ret_res = self.lower_res(resolutions.next().unwrap_or(Res::Err));

                // Here, we are looping over namespaces, if they exist for the definition
                // being imported. We only handle type and value namespaces because we
                // won't be dealing with macros in the rest of the compiler.
                // Essentially a single `use` which imports two names is desugared into
                // two imports.
                for (res, &new_node_id) in resolutions.zip([id1, id2].iter()) {
                    let ident = *ident;
                    let mut path = path.clone();
                    for seg in &mut path.segments {
                        seg.id = self.sess.next_node_id();
                    }
                    let span = path.span;

                    self.with_hir_id_owner(new_node_id, |this| {
                        let new_id = this.lower_node_id(new_node_id);
                        let res = this.lower_res(res);
                        let path =
                            this.lower_path_extra(res, &path, ParamMode::Explicit, None);
                        let item = hir::ItemKind::Use(P(path), hir::UseKind::Single);
                        let vis = this.rebuild_vis(&vis);

                        this.insert_item(
                            hir::Item {
                                hir_id: new_id,
                                ident,
                                attrs: attrs.into_iter().cloned().collect(),
                                node: item,
                                vis,
                                span,
                            },
                        );
                    });
                }

                let path =
                    P(self.lower_path_extra(ret_res, &path, ParamMode::Explicit, None));
                hir::ItemKind::Use(path, hir::UseKind::Single)
            }
            UseTreeKind::Glob => {
                let path = P(self.lower_path(
                    id,
                    &Path {
                        segments,
                        span: path.span,
                    },
                    ParamMode::Explicit,
                ));
                hir::ItemKind::Use(path, hir::UseKind::Glob)
            }
            UseTreeKind::Nested(ref trees) => {
                // Nested imports are desugared into simple imports.
                // So, if we start with
                //
                // ```
                // pub(x) use foo::{a, b};
                // ```
                //
                // we will create three items:
                //
                // ```
                // pub(x) use foo::a;
                // pub(x) use foo::b;
                // pub(x) use foo::{}; // <-- this is called the `ListStem`
                // ```
                //
                // The first two are produced by recursively invoking
                // `lower_use_tree` (and indeed there may be things
                // like `use foo::{a::{b, c}}` and so forth).  They
                // wind up being directly added to
                // `self.items`. However, the structure of this
                // function also requires us to return one item, and
                // for that we return the `{}` import (called the
                // `ListStem`).

                let prefix = Path {
                    segments,
                    span: prefix.span.to(path.span),
                };

                // Add all the nested `PathListItem`s to the HIR.
                for &(ref use_tree, id) in trees {
                    let new_hir_id = self.lower_node_id(id);

                    let mut prefix = prefix.clone();

                    // Give the segments new node-ids since they are being cloned.
                    for seg in &mut prefix.segments {
                        seg.id = self.sess.next_node_id();
                    }

                    // Each `use` import is an item and thus are owners of the
                    // names in the path. Up to this point the nested import is
                    // the current owner, since we want each desugared import to
                    // own its own names, we have to adjust the owner before
                    // lowering the rest of the import.
                    self.with_hir_id_owner(id, |this| {
                        let mut vis = this.rebuild_vis(&vis);
                        let mut ident = *ident;

                        let item = this.lower_use_tree(use_tree,
                                                       &prefix,
                                                       id,
                                                       &mut vis,
                                                       &mut ident,
                                                       attrs);

                        this.insert_item(
                            hir::Item {
                                hir_id: new_hir_id,
                                ident,
                                attrs: attrs.into_iter().cloned().collect(),
                                node: item,
                                vis,
                                span: use_tree.span,
                            },
                        );
                    });
                }

                // Subtle and a bit hacky: we lower the privacy level
                // of the list stem to "private" most of the time, but
                // not for "restricted" paths. The key thing is that
                // we don't want it to stay as `pub` (with no caveats)
                // because that affects rustdoc and also the lints
                // about `pub` items. But we can't *always* make it
                // private -- particularly not for restricted paths --
                // because it contains node-ids that would then be
                // unused, failing the check that HirIds are "densely
                // assigned".
                match vis.node {
                    hir::VisibilityKind::Public |
                    hir::VisibilityKind::Crate(_) |
                    hir::VisibilityKind::Inherited => {
                        *vis = respan(prefix.span.shrink_to_lo(), hir::VisibilityKind::Inherited);
                    }
                    hir::VisibilityKind::Restricted { .. } => {
                        // Do nothing here, as described in the comment on the match.
                    }
                }

                let res = self.expect_full_res_from_use(id).next().unwrap_or(Res::Err);
                let res = self.lower_res(res);
                let path = P(self.lower_path_extra(res, &prefix, ParamMode::Explicit, None));
                hir::ItemKind::Use(path, hir::UseKind::ListStem)
            }
        }
    }

    /// Paths like the visibility path in `pub(super) use foo::{bar, baz}` are repeated
    /// many times in the HIR tree; for each occurrence, we need to assign distinct
    /// `NodeId`s. (See, e.g., #56128.)
    fn rebuild_use_path(&mut self, path: &hir::Path) -> hir::Path {
        debug!("rebuild_use_path(path = {:?})", path);
        let segments = path.segments.iter().map(|seg| hir::PathSegment {
            ident: seg.ident,
            hir_id: seg.hir_id.map(|_| self.next_id()),
            res: seg.res,
            args: None,
            infer_args: seg.infer_args,
        }).collect();
        hir::Path {
            span: path.span,
            res: path.res,
            segments,
        }
    }

    fn rebuild_vis(&mut self, vis: &hir::Visibility) -> hir::Visibility {
        let vis_kind = match vis.node {
            hir::VisibilityKind::Public => hir::VisibilityKind::Public,
            hir::VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            hir::VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
            hir::VisibilityKind::Restricted { ref path, hir_id: _ } => {
                hir::VisibilityKind::Restricted {
                    path: P(self.rebuild_use_path(path)),
                    hir_id: self.next_id(),
                }
            }
        };
        respan(vis.span, vis_kind)
    }

    fn lower_trait_item(&mut self, i: &TraitItem) -> hir::TraitItem {
        let trait_item_def_id = self.resolver.definitions().local_def_id(i.id);

        let (generics, node) = match i.node {
            TraitItemKind::Const(ref ty, ref default) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::TraitItemKind::Const(
                    self.lower_ty(ty, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_const_body(x)),
                ),
            ),
            TraitItemKind::Method(ref sig, None) => {
                let names = self.lower_fn_args_to_names(&sig.decl);
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    trait_item_def_id,
                    false,
                    None,
                );
                (generics, hir::TraitItemKind::Method(sig, hir::TraitMethod::Required(names)))
            }
            TraitItemKind::Method(ref sig, Some(ref body)) => {
                let body_id = self.lower_fn_body(&sig.decl, |this| {
                    let body = this.lower_block(body, false);
                    this.expr_block(body, ThinVec::new())
                });
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    trait_item_def_id,
                    false,
                    None,
                );
                (generics, hir::TraitItemKind::Method(sig, hir::TraitMethod::Provided(body_id)))
            }
            TraitItemKind::Type(ref bounds, ref default) => {
                let generics = self.lower_generics(&i.generics, ImplTraitContext::disallowed());
                let node = hir::TraitItemKind::Type(
                    self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_ty(x, ImplTraitContext::disallowed())),
                );

                (generics, node)
            },
            TraitItemKind::Macro(..) => bug!("macro item shouldn't exist at this point"),
        };

        hir::TraitItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            generics,
            node,
            span: i.span,
        }
    }

    fn lower_trait_item_ref(&mut self, i: &TraitItem) -> hir::TraitItemRef {
        let (kind, has_default) = match i.node {
            TraitItemKind::Const(_, ref default) => {
                (hir::AssocItemKind::Const, default.is_some())
            }
            TraitItemKind::Type(_, ref default) => {
                (hir::AssocItemKind::Type, default.is_some())
            }
            TraitItemKind::Method(ref sig, ref default) => (
                hir::AssocItemKind::Method {
                    has_self: sig.decl.has_self(),
                },
                default.is_some(),
            ),
            TraitItemKind::Macro(..) => unimplemented!(),
        };
        hir::TraitItemRef {
            id: hir::TraitItemId { hir_id: self.lower_node_id(i.id) },
            ident: i.ident,
            span: i.span,
            defaultness: self.lower_defaultness(Defaultness::Default, has_default),
            kind,
        }
    }

    fn lower_impl_item(&mut self, i: &ImplItem) -> hir::ImplItem {
        let impl_item_def_id = self.resolver.definitions().local_def_id(i.id);

        let (generics, node) = match i.node {
            ImplItemKind::Const(ref ty, ref expr) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::Const(
                    self.lower_ty(ty, ImplTraitContext::disallowed()),
                    self.lower_const_body(expr),
                ),
            ),
            ImplItemKind::Method(ref sig, ref body) => {
                self.current_item = Some(i.span);
                let body_id = self.lower_maybe_async_body(
                    &sig.decl, sig.header.asyncness.node, body
                );
                let impl_trait_return_allow = !self.is_in_trait_impl;
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    impl_item_def_id,
                    impl_trait_return_allow,
                    sig.header.asyncness.node.opt_return_id(),
                );

                (generics, hir::ImplItemKind::Method(sig, body_id))
            }
            ImplItemKind::Type(ref ty) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::Type(self.lower_ty(ty, ImplTraitContext::disallowed())),
            ),
            ImplItemKind::Existential(ref bounds) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::Existential(
                    self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
                ),
            ),
            ImplItemKind::Macro(..) => bug!("`TyMac` should have been expanded by now"),
        };

        hir::ImplItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            generics,
            vis: self.lower_visibility(&i.vis, None),
            defaultness: self.lower_defaultness(i.defaultness, true /* [1] */),
            node,
            span: i.span,
        }

        // [1] since `default impl` is not yet implemented, this is always true in impls
    }

    fn lower_impl_item_ref(&mut self, i: &ImplItem) -> hir::ImplItemRef {
        hir::ImplItemRef {
            id: hir::ImplItemId { hir_id: self.lower_node_id(i.id) },
            ident: i.ident,
            span: i.span,
            vis: self.lower_visibility(&i.vis, Some(i.id)),
            defaultness: self.lower_defaultness(i.defaultness, true /* [1] */),
            kind: match i.node {
                ImplItemKind::Const(..) => hir::AssocItemKind::Const,
                ImplItemKind::Type(..) => hir::AssocItemKind::Type,
                ImplItemKind::Existential(..) => hir::AssocItemKind::Existential,
                ImplItemKind::Method(ref sig, _) => hir::AssocItemKind::Method {
                    has_self: sig.decl.has_self(),
                },
                ImplItemKind::Macro(..) => unimplemented!(),
            },
        }

        // [1] since `default impl` is not yet implemented, this is always true in impls
    }

    fn lower_mod(&mut self, m: &Mod) -> hir::Mod {
        hir::Mod {
            inner: m.inner,
            item_ids: m.items.iter().flat_map(|x| self.lower_item_id(x)).collect(),
        }
    }

    fn lower_item_id(&mut self, i: &Item) -> SmallVec<[hir::ItemId; 1]> {
        let node_ids = match i.node {
            ItemKind::Use(ref use_tree) => {
                let mut vec = smallvec![i.id];
                self.lower_item_id_use_tree(use_tree, i.id, &mut vec);
                vec
            }
            ItemKind::MacroDef(..) => SmallVec::new(),
            ItemKind::Fn(..) |
            ItemKind::Impl(.., None, _, _) => smallvec![i.id],
            ItemKind::Static(ref ty, ..) => {
                let mut ids = smallvec![i.id];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            ItemKind::Const(ref ty, ..) => {
                let mut ids = smallvec![i.id];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            _ => smallvec![i.id],
        };

        node_ids.into_iter().map(|node_id| hir::ItemId {
            id: self.allocate_hir_id_counter(node_id)
        }).collect()
    }

    fn lower_item_id_use_tree(&mut self,
                              tree: &UseTree,
                              base_id: NodeId,
                              vec: &mut SmallVec<[NodeId; 1]>)
    {
        match tree.kind {
            UseTreeKind::Nested(ref nested_vec) => for &(ref nested, id) in nested_vec {
                vec.push(id);
                self.lower_item_id_use_tree(nested, id, vec);
            },
            UseTreeKind::Glob => {}
            UseTreeKind::Simple(_, id1, id2) => {
                for (_, &id) in self.expect_full_res_from_use(base_id)
                                    .skip(1)
                                    .zip([id1, id2].iter())
                {
                    vec.push(id);
                }
            },
        }
    }

    pub fn lower_item(&mut self, i: &Item) -> Option<hir::Item> {
        let mut ident = i.ident;
        let mut vis = self.lower_visibility(&i.vis, None);
        let attrs = self.lower_attrs(&i.attrs);
        if let ItemKind::MacroDef(ref def) = i.node {
            if !def.legacy || attr::contains_name(&i.attrs, sym::macro_export) ||
                              attr::contains_name(&i.attrs, sym::rustc_doc_only_macro) {
                let body = self.lower_token_stream(def.stream());
                let hir_id = self.lower_node_id(i.id);
                self.exported_macros.push(hir::MacroDef {
                    name: ident.name,
                    vis,
                    attrs,
                    hir_id,
                    span: i.span,
                    body,
                    legacy: def.legacy,
                });
            }
            return None;
        }

        let node = self.lower_item_kind(i.id, &mut ident, &attrs, &mut vis, &i.node);

        Some(hir::Item {
            hir_id: self.lower_node_id(i.id),
            ident,
            attrs,
            node,
            vis,
            span: i.span,
        })
    }

    fn lower_foreign_item(&mut self, i: &ForeignItem) -> hir::ForeignItem {
        let def_id = self.resolver.definitions().local_def_id(i.id);
        hir::ForeignItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            node: match i.node {
                ForeignItemKind::Fn(ref fdec, ref generics) => {
                    let (generics, (fn_dec, fn_args)) = self.add_in_band_defs(
                        generics,
                        def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, _| {
                            (
                                // Disallow impl Trait in foreign items
                                this.lower_fn_decl(fdec, None, false, None),
                                this.lower_fn_args_to_names(fdec),
                            )
                        },
                    );

                    hir::ForeignItemKind::Fn(fn_dec, fn_args, generics)
                }
                ForeignItemKind::Static(ref t, m) => {
                    hir::ForeignItemKind::Static(
                        self.lower_ty(t, ImplTraitContext::disallowed()), self.lower_mutability(m))
                }
                ForeignItemKind::Ty => hir::ForeignItemKind::Type,
                ForeignItemKind::Macro(_) => panic!("shouldn't exist here"),
            },
            vis: self.lower_visibility(&i.vis, None),
            span: i.span,
        }
    }

    fn lower_method_sig(
        &mut self,
        generics: &Generics,
        sig: &MethodSig,
        fn_def_id: DefId,
        impl_trait_return_allow: bool,
        is_async: Option<NodeId>,
    ) -> (hir::Generics, hir::MethodSig) {
        let header = self.lower_fn_header(sig.header);
        let (generics, decl) = self.add_in_band_defs(
            generics,
            fn_def_id,
            AnonymousLifetimeMode::PassThrough,
            |this, idty| this.lower_fn_decl(
                &sig.decl,
                Some((fn_def_id, idty)),
                impl_trait_return_allow,
                is_async,
            ),
        );
        (generics, hir::MethodSig { header, decl })
    }

    fn lower_is_auto(&mut self, a: IsAuto) -> hir::IsAuto {
        match a {
            IsAuto::Yes => hir::IsAuto::Yes,
            IsAuto::No => hir::IsAuto::No,
        }
    }

    fn lower_fn_header(&mut self, h: FnHeader) -> hir::FnHeader {
        hir::FnHeader {
            unsafety: self.lower_unsafety(h.unsafety),
            asyncness: self.lower_asyncness(h.asyncness.node),
            constness: self.lower_constness(h.constness),
            abi: h.abi,
        }
    }

    fn lower_unsafety(&mut self, u: Unsafety) -> hir::Unsafety {
        match u {
            Unsafety::Unsafe => hir::Unsafety::Unsafe,
            Unsafety::Normal => hir::Unsafety::Normal,
        }
    }

    fn lower_constness(&mut self, c: Spanned<Constness>) -> hir::Constness {
        match c.node {
            Constness::Const => hir::Constness::Const,
            Constness::NotConst => hir::Constness::NotConst,
        }
    }

    fn lower_asyncness(&mut self, a: IsAsync) -> hir::IsAsync {
        match a {
            IsAsync::Async { .. } => hir::IsAsync::Async,
            IsAsync::NotAsync => hir::IsAsync::NotAsync,
        }
    }

    fn lower_unop(&mut self, u: UnOp) -> hir::UnOp {
        match u {
            UnOp::Deref => hir::UnDeref,
            UnOp::Not => hir::UnNot,
            UnOp::Neg => hir::UnNeg,
        }
    }

    fn lower_binop(&mut self, b: BinOp) -> hir::BinOp {
        Spanned {
            node: match b.node {
                BinOpKind::Add => hir::BinOpKind::Add,
                BinOpKind::Sub => hir::BinOpKind::Sub,
                BinOpKind::Mul => hir::BinOpKind::Mul,
                BinOpKind::Div => hir::BinOpKind::Div,
                BinOpKind::Rem => hir::BinOpKind::Rem,
                BinOpKind::And => hir::BinOpKind::And,
                BinOpKind::Or => hir::BinOpKind::Or,
                BinOpKind::BitXor => hir::BinOpKind::BitXor,
                BinOpKind::BitAnd => hir::BinOpKind::BitAnd,
                BinOpKind::BitOr => hir::BinOpKind::BitOr,
                BinOpKind::Shl => hir::BinOpKind::Shl,
                BinOpKind::Shr => hir::BinOpKind::Shr,
                BinOpKind::Eq => hir::BinOpKind::Eq,
                BinOpKind::Lt => hir::BinOpKind::Lt,
                BinOpKind::Le => hir::BinOpKind::Le,
                BinOpKind::Ne => hir::BinOpKind::Ne,
                BinOpKind::Ge => hir::BinOpKind::Ge,
                BinOpKind::Gt => hir::BinOpKind::Gt,
            },
            span: b.span,
        }
    }

    fn lower_pat(&mut self, p: &Pat) -> P<hir::Pat> {
        let node = match p.node {
            PatKind::Wild => hir::PatKind::Wild,
            PatKind::Ident(ref binding_mode, ident, ref sub) => {
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
                            sub.as_ref().map(|x| self.lower_pat(x)),
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
            PatKind::Lit(ref e) => hir::PatKind::Lit(P(self.lower_expr(e))),
            PatKind::TupleStruct(ref path, ref pats, ddpos) => {
                let qpath = self.lower_qpath(
                    p.id,
                    &None,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );
                hir::PatKind::TupleStruct(
                    qpath,
                    pats.iter().map(|x| self.lower_pat(x)).collect(),
                    ddpos,
                )
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
                    .map(|f| {
                        Spanned {
                            span: f.span,
                            node: hir::FieldPat {
                                hir_id: self.next_id(),
                                ident: f.node.ident,
                                pat: self.lower_pat(&f.node.pat),
                                is_shorthand: f.node.is_shorthand,
                            },
                        }
                    })
                    .collect();
                hir::PatKind::Struct(qpath, fs, etc)
            }
            PatKind::Tuple(ref elts, ddpos) => {
                hir::PatKind::Tuple(elts.iter().map(|x| self.lower_pat(x)).collect(), ddpos)
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
            PatKind::Slice(ref before, ref slice, ref after) => hir::PatKind::Slice(
                before.iter().map(|x| self.lower_pat(x)).collect(),
                slice.as_ref().map(|x| self.lower_pat(x)),
                after.iter().map(|x| self.lower_pat(x)).collect(),
            ),
            PatKind::Paren(ref inner) => return self.lower_pat(inner),
            PatKind::Mac(_) => panic!("Shouldn't exist here"),
        };

        P(hir::Pat {
            hir_id: self.lower_node_id(p.id),
            node,
            span: p.span,
        })
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

    fn lower_expr(&mut self, e: &Expr) -> hir::Expr {
        let kind = match e.node {
            ExprKind::Box(ref inner) => hir::ExprKind::Box(P(self.lower_expr(inner))),
            ExprKind::Array(ref exprs) => {
                hir::ExprKind::Array(exprs.iter().map(|x| self.lower_expr(x)).collect())
            }
            ExprKind::Repeat(ref expr, ref count) => {
                let expr = P(self.lower_expr(expr));
                let count = self.lower_anon_const(count);
                hir::ExprKind::Repeat(expr, count)
            }
            ExprKind::Tup(ref elts) => {
                hir::ExprKind::Tup(elts.iter().map(|x| self.lower_expr(x)).collect())
            }
            ExprKind::Call(ref f, ref args) => {
                let f = P(self.lower_expr(f));
                hir::ExprKind::Call(f, args.iter().map(|x| self.lower_expr(x)).collect())
            }
            ExprKind::MethodCall(ref seg, ref args) => {
                let hir_seg = P(self.lower_path_segment(
                    e.span,
                    seg,
                    ParamMode::Optional,
                    0,
                    ParenthesizedGenericArgs::Err,
                    ImplTraitContext::disallowed(),
                    None,
                ));
                let args = args.iter().map(|x| self.lower_expr(x)).collect();
                hir::ExprKind::MethodCall(hir_seg, seg.ident.span, args)
            }
            ExprKind::Binary(binop, ref lhs, ref rhs) => {
                let binop = self.lower_binop(binop);
                let lhs = P(self.lower_expr(lhs));
                let rhs = P(self.lower_expr(rhs));
                hir::ExprKind::Binary(binop, lhs, rhs)
            }
            ExprKind::Unary(op, ref ohs) => {
                let op = self.lower_unop(op);
                let ohs = P(self.lower_expr(ohs));
                hir::ExprKind::Unary(op, ohs)
            }
            ExprKind::Lit(ref l) => hir::ExprKind::Lit(respan(l.span, l.node.clone())),
            ExprKind::Cast(ref expr, ref ty) => {
                let expr = P(self.lower_expr(expr));
                hir::ExprKind::Cast(expr, self.lower_ty(ty, ImplTraitContext::disallowed()))
            }
            ExprKind::Type(ref expr, ref ty) => {
                let expr = P(self.lower_expr(expr));
                hir::ExprKind::Type(expr, self.lower_ty(ty, ImplTraitContext::disallowed()))
            }
            ExprKind::AddrOf(m, ref ohs) => {
                let m = self.lower_mutability(m);
                let ohs = P(self.lower_expr(ohs));
                hir::ExprKind::AddrOf(m, ohs)
            }
            ExprKind::Let(ref pats, ref scrutinee) => {
                // If we got here, the `let` expression is not allowed.
                self.sess
                    .struct_span_err(e.span, "`let` expressions are not supported here")
                    .note("only supported directly in conditions of `if`- and `while`-expressions")
                    .note("as well as when nested within `&&` and parenthesis in those conditions")
                    .emit();

                // For better recovery, we emit:
                // ```
                // match scrutinee { pats => true, _ => false }
                // ```
                // While this doesn't fully match the user's intent, it has key advantages:
                // 1. We can avoid using `abort_if_errors`.
                // 2. We can typeck both `pats` and `scrutinee`.
                // 3. `pats` is allowed to be refutable.
                // 4. The return type of the block is `bool` which seems like what the user wanted.
                let scrutinee = self.lower_expr(scrutinee);
                let then_arm = {
                    let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                    let expr = self.expr_bool(e.span, true);
                    self.arm(pats, P(expr))
                };
                let else_arm = {
                    let pats = hir_vec![self.pat_wild(e.span)];
                    let expr = self.expr_bool(e.span, false);
                    self.arm(pats, P(expr))
                };
                hir::ExprKind::Match(
                    P(scrutinee),
                    vec![then_arm, else_arm].into(),
                    hir::MatchSource::Normal,
                )
            }
            // FIXME(#53667): handle lowering of && and parens.
            ExprKind::If(ref cond, ref then, ref else_opt) => {
                // `_ => else_block` where `else_block` is `{}` if there's `None`:
                let else_pat = self.pat_wild(e.span);
                let (else_expr, contains_else_clause) = match else_opt {
                    None => (self.expr_block_empty(e.span), false),
                    Some(els) => (self.lower_expr(els), true),
                };
                let else_arm = self.arm(hir_vec![else_pat], P(else_expr));

                // Handle then + scrutinee:
                let then_blk = self.lower_block(then, false);
                let then_expr = self.expr_block(then_blk, ThinVec::new());
                let (then_pats, scrutinee, desugar) = match cond.node {
                    // `<pat> => <then>`
                    ExprKind::Let(ref pats, ref scrutinee) => {
                        let scrutinee = self.lower_expr(scrutinee);
                        let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                        let desugar = hir::MatchSource::IfLetDesugar { contains_else_clause };
                        (pats, scrutinee, desugar)
                    }
                    // `true => then`:
                    _ => {
                        // Lower condition:
                        let cond = self.lower_expr(cond);
                        // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                        // to preserve drop semantics since `if cond { ... }`
                        // don't let temporaries live outside of `cond`.
                        let span_block = self.mark_span_with_reason(IfTemporary, cond.span, None);
                        // Wrap in a construct equivalent to `{ let _t = $cond; _t }`
                        // to preserve drop semantics since `if cond { ... }` does not
                        // let temporaries live outside of `cond`.
                        let cond = self.expr_drop_temps(span_block, P(cond), ThinVec::new());

                        let desugar = hir::MatchSource::IfDesugar { contains_else_clause };
                        let pats = hir_vec![self.pat_bool(e.span, true)];
                        (pats, cond, desugar)
                    }
                };
                let then_arm = self.arm(then_pats, P(then_expr));

                hir::ExprKind::Match(P(scrutinee), vec![then_arm, else_arm].into(), desugar)
            }
            // FIXME(#53667): handle lowering of && and parens.
            ExprKind::While(ref cond, ref body, opt_label) => {
                // Desugar `ExprWhileLet`
                // from: `[opt_ident]: while let <pat> = <sub_expr> <body>`
                if let ExprKind::Let(ref pats, ref sub_expr) = cond.node {
                    // to:
                    //
                    //   [opt_ident]: loop {
                    //     match <sub_expr> {
                    //       <pat> => <body>,
                    //       _ => break
                    //     }
                    //   }

                    // Note that the block AND the condition are evaluated in the loop scope.
                    // This is done to allow `break` from inside the condition of the loop.
                    let (body, break_expr, sub_expr) = self.with_loop_scope(e.id, |this| {
                        (
                            this.lower_block(body, false),
                            this.expr_break(e.span, ThinVec::new()),
                            this.with_loop_condition_scope(|this| P(this.lower_expr(sub_expr))),
                        )
                    });

                    // `<pat> => <body>`
                    let pat_arm = {
                        let body_expr = P(self.expr_block(body, ThinVec::new()));
                        let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                        self.arm(pats, body_expr)
                    };

                    // `_ => break`
                    let break_arm = {
                        let pat_under = self.pat_wild(e.span);
                        self.arm(hir_vec![pat_under], break_expr)
                    };

                    // `match <sub_expr> { ... }`
                    let arms = hir_vec![pat_arm, break_arm];
                    let match_expr = self.expr(
                        sub_expr.span,
                        hir::ExprKind::Match(sub_expr, arms, hir::MatchSource::WhileLetDesugar),
                        ThinVec::new(),
                    );

                    // `[opt_ident]: loop { ... }`
                    let loop_block = P(self.block_expr(P(match_expr)));
                    let loop_expr = hir::ExprKind::Loop(
                        loop_block,
                        self.lower_label(opt_label),
                        hir::LoopSource::WhileLet,
                    );
                    // Add attributes to the outer returned expr node.
                    loop_expr
                } else {
                    self.with_loop_scope(e.id, |this| {
                        hir::ExprKind::While(
                            this.with_loop_condition_scope(|this| P(this.lower_expr(cond))),
                            this.lower_block(body, false),
                            this.lower_label(opt_label),
                        )
                    })
                }
            }
            ExprKind::Loop(ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                hir::ExprKind::Loop(
                    this.lower_block(body, false),
                    this.lower_label(opt_label),
                    hir::LoopSource::Loop,
                )
            }),
            ExprKind::TryBlock(ref body) => {
                self.with_catch_scope(body.id, |this| {
                    let unstable_span = this.mark_span_with_reason(
                        CompilerDesugaringKind::TryBlock,
                        body.span,
                        this.allow_try_trait.clone(),
                    );
                    let mut block = this.lower_block(body, true).into_inner();
                    let tail = block.expr.take().map_or_else(
                        || {
                            let span = this.sess.source_map().end_point(unstable_span);
                            hir::Expr {
                                span,
                                node: hir::ExprKind::Tup(hir_vec![]),
                                attrs: ThinVec::new(),
                                hir_id: this.next_id(),
                            }
                        },
                        |x: P<hir::Expr>| x.into_inner(),
                    );
                    block.expr = Some(this.wrap_in_try_constructor(
                        sym::from_ok, tail, unstable_span));
                    hir::ExprKind::Block(P(block), None)
                })
            }
            ExprKind::Match(ref expr, ref arms) => hir::ExprKind::Match(
                P(self.lower_expr(expr)),
                arms.iter().map(|x| self.lower_arm(x)).collect(),
                hir::MatchSource::Normal,
            ),
            ExprKind::Async(capture_clause, closure_node_id, ref block) => {
                self.make_async_expr(capture_clause, closure_node_id, None, block.span, |this| {
                    this.with_new_scopes(|this| {
                        let block = this.lower_block(block, false);
                        this.expr_block(block, ThinVec::new())
                    })
                })
            }
            ExprKind::Await(_origin, ref expr) => self.lower_await(e.span, expr),
            ExprKind::Closure(
                capture_clause, asyncness, movability, ref decl, ref body, fn_decl_span
            ) => {
                if let IsAsync::Async { closure_id, .. } = asyncness {
                    let outer_decl = FnDecl {
                        inputs: decl.inputs.clone(),
                        output: FunctionRetTy::Default(fn_decl_span),
                        c_variadic: false,
                    };
                    // We need to lower the declaration outside the new scope, because we
                    // have to conserve the state of being inside a loop condition for the
                    // closure argument types.
                    let fn_decl = self.lower_fn_decl(&outer_decl, None, false, None);

                    self.with_new_scopes(|this| {
                        // FIXME(cramertj): allow `async` non-`move` closures with arguments.
                        if capture_clause == CaptureBy::Ref &&
                            !decl.inputs.is_empty()
                        {
                            struct_span_err!(
                                this.sess,
                                fn_decl_span,
                                E0708,
                                "`async` non-`move` closures with arguments \
                                are not currently supported",
                            )
                                .help("consider using `let` statements to manually capture \
                                       variables by reference before entering an \
                                       `async move` closure")
                                .emit();
                        }

                        // Transform `async |x: u8| -> X { ... }` into
                        // `|x: u8| future_from_generator(|| -> X { ... })`.
                        let body_id = this.lower_fn_body(&outer_decl, |this| {
                            let async_ret_ty = if let FunctionRetTy::Ty(ty) = &decl.output {
                                Some(ty.clone())
                            } else { None };
                            let async_body = this.make_async_expr(
                                capture_clause, closure_id, async_ret_ty, body.span,
                                |this| {
                                    this.with_new_scopes(|this| this.lower_expr(body))
                                });
                            this.expr(fn_decl_span, async_body, ThinVec::new())
                        });
                        hir::ExprKind::Closure(
                            this.lower_capture_clause(capture_clause),
                            fn_decl,
                            body_id,
                            fn_decl_span,
                            None,
                        )
                    })
                } else {
                    // Lower outside new scope to preserve `is_in_loop_condition`.
                    let fn_decl = self.lower_fn_decl(decl, None, false, None);

                    self.with_new_scopes(|this| {
                        this.current_item = Some(fn_decl_span);
                        let mut generator_kind = None;
                        let body_id = this.lower_fn_body(decl, |this| {
                            let e = this.lower_expr(body);
                            generator_kind = this.generator_kind;
                            e
                        });
                        let generator_option = this.generator_movability_for_fn(
                            &decl,
                            fn_decl_span,
                            generator_kind,
                            movability,
                        );
                        hir::ExprKind::Closure(
                            this.lower_capture_clause(capture_clause),
                            fn_decl,
                            body_id,
                            fn_decl_span,
                            generator_option,
                        )
                    })
                }
            }
            ExprKind::Block(ref blk, opt_label) => {
                hir::ExprKind::Block(self.lower_block(blk,
                                                      opt_label.is_some()),
                                                      self.lower_label(opt_label))
            }
            ExprKind::Assign(ref el, ref er) => {
                hir::ExprKind::Assign(P(self.lower_expr(el)), P(self.lower_expr(er)))
            }
            ExprKind::AssignOp(op, ref el, ref er) => hir::ExprKind::AssignOp(
                self.lower_binop(op),
                P(self.lower_expr(el)),
                P(self.lower_expr(er)),
            ),
            ExprKind::Field(ref el, ident) => hir::ExprKind::Field(P(self.lower_expr(el)), ident),
            ExprKind::Index(ref el, ref er) => {
                hir::ExprKind::Index(P(self.lower_expr(el)), P(self.lower_expr(er)))
            }
            // Desugar `<start>..=<end>` into `std::ops::RangeInclusive::new(<start>, <end>)`.
            ExprKind::Range(Some(ref e1), Some(ref e2), RangeLimits::Closed) => {
                let id = self.next_id();
                let e1 = self.lower_expr(e1);
                let e2 = self.lower_expr(e2);
                self.expr_call_std_assoc_fn(
                    id,
                    e.span,
                    &[sym::ops, sym::RangeInclusive],
                    "new",
                    hir_vec![e1, e2],
                )
            }
            ExprKind::Range(ref e1, ref e2, lims) => {
                use syntax::ast::RangeLimits::*;

                let path = match (e1, e2, lims) {
                    (&None, &None, HalfOpen) => sym::RangeFull,
                    (&Some(..), &None, HalfOpen) => sym::RangeFrom,
                    (&None, &Some(..), HalfOpen) => sym::RangeTo,
                    (&Some(..), &Some(..), HalfOpen) => sym::Range,
                    (&None, &Some(..), Closed) => sym::RangeToInclusive,
                    (&Some(..), &Some(..), Closed) => unreachable!(),
                    (_, &None, Closed) => self.diagnostic()
                        .span_fatal(e.span, "inclusive range with no end")
                        .raise(),
                };

                let fields = e1.iter()
                    .map(|e| ("start", e))
                    .chain(e2.iter().map(|e| ("end", e)))
                    .map(|(s, e)| {
                        let expr = P(self.lower_expr(&e));
                        let ident = Ident::new(Symbol::intern(s), e.span);
                        self.field(ident, expr, e.span)
                    })
                    .collect::<P<[hir::Field]>>();

                let is_unit = fields.is_empty();
                let struct_path = [sym::ops, path];
                let struct_path = self.std_path(e.span, &struct_path, None, is_unit);
                let struct_path = hir::QPath::Resolved(None, P(struct_path));

                return hir::Expr {
                    hir_id: self.lower_node_id(e.id),
                    node: if is_unit {
                        hir::ExprKind::Path(struct_path)
                    } else {
                        hir::ExprKind::Struct(P(struct_path), fields, None)
                    },
                    span: e.span,
                    attrs: e.attrs.clone(),
                };
            }
            ExprKind::Path(ref qself, ref path) => {
                let qpath = self.lower_qpath(
                    e.id,
                    qself,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                );
                hir::ExprKind::Path(qpath)
            }
            ExprKind::Break(opt_label, ref opt_expr) => {
                let destination = if self.is_in_loop_condition && opt_label.is_none() {
                    hir::Destination {
                        label: None,
                        target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition).into(),
                    }
                } else {
                    self.lower_loop_destination(opt_label.map(|label| (e.id, label)))
                };
                hir::ExprKind::Break(
                    destination,
                    opt_expr.as_ref().map(|x| P(self.lower_expr(x))),
                )
            }
            ExprKind::Continue(opt_label) => {
                hir::ExprKind::Continue(if self.is_in_loop_condition && opt_label.is_none() {
                    hir::Destination {
                        label: None,
                        target_id: Err(hir::LoopIdError::UnlabeledCfInWhileCondition).into(),
                    }
                } else {
                    self.lower_loop_destination(opt_label.map(|label| (e.id, label)))
                })
            }
            ExprKind::Ret(ref e) => hir::ExprKind::Ret(e.as_ref().map(|x| P(self.lower_expr(x)))),
            ExprKind::InlineAsm(ref asm) => {
                let hir_asm = hir::InlineAsm {
                    inputs: asm.inputs.iter().map(|&(ref c, _)| c.clone()).collect(),
                    outputs: asm.outputs
                        .iter()
                        .map(|out| hir::InlineAsmOutput {
                            constraint: out.constraint.clone(),
                            is_rw: out.is_rw,
                            is_indirect: out.is_indirect,
                            span: out.expr.span,
                        })
                        .collect(),
                    asm: asm.asm.clone(),
                    asm_str_style: asm.asm_str_style,
                    clobbers: asm.clobbers.clone().into(),
                    volatile: asm.volatile,
                    alignstack: asm.alignstack,
                    dialect: asm.dialect,
                    ctxt: asm.ctxt,
                };
                let outputs = asm.outputs
                    .iter()
                    .map(|out| self.lower_expr(&out.expr))
                    .collect();
                let inputs = asm.inputs
                    .iter()
                    .map(|&(_, ref input)| self.lower_expr(input))
                    .collect();
                hir::ExprKind::InlineAsm(P(hir_asm), outputs, inputs)
            }
            ExprKind::Struct(ref path, ref fields, ref maybe_expr) => hir::ExprKind::Struct(
                P(self.lower_qpath(
                    e.id,
                    &None,
                    path,
                    ParamMode::Optional,
                    ImplTraitContext::disallowed(),
                )),
                fields.iter().map(|x| self.lower_field(x)).collect(),
                maybe_expr.as_ref().map(|x| P(self.lower_expr(x))),
            ),
            ExprKind::Paren(ref ex) => {
                let mut ex = self.lower_expr(ex);
                // Include parens in span, but only if it is a super-span.
                if e.span.contains(ex.span) {
                    ex.span = e.span;
                }
                // Merge attributes into the inner expression.
                let mut attrs = e.attrs.clone();
                attrs.extend::<Vec<_>>(ex.attrs.into());
                ex.attrs = attrs;
                return ex;
            }

            ExprKind::Yield(ref opt_expr) => {
                match self.generator_kind {
                    Some(hir::GeneratorKind::Gen) => {},
                    Some(hir::GeneratorKind::Async) => {
                        span_err!(
                            self.sess,
                            e.span,
                            E0727,
                            "`async` generators are not yet supported",
                        );
                        self.sess.abort_if_errors();
                    },
                    None => {
                        self.generator_kind = Some(hir::GeneratorKind::Gen);
                    }
                }
                let expr = opt_expr
                    .as_ref()
                    .map(|x| self.lower_expr(x))
                    .unwrap_or_else(|| self.expr_unit(e.span));
                hir::ExprKind::Yield(P(expr), hir::YieldSource::Yield)
            }

            ExprKind::Err => hir::ExprKind::Err,

            // Desugar `ExprForLoop`
            // from: `[opt_ident]: for <pat> in <head> <body>`
            ExprKind::ForLoop(ref pat, ref head, ref body, opt_label) => {
                // to:
                //
                //   {
                //     let result = match ::std::iter::IntoIterator::into_iter(<head>) {
                //       mut iter => {
                //         [opt_ident]: loop {
                //           let mut __next;
                //           match ::std::iter::Iterator::next(&mut iter) {
                //             ::std::option::Option::Some(val) => __next = val,
                //             ::std::option::Option::None => break
                //           };
                //           let <pat> = __next;
                //           StmtKind::Expr(<body>);
                //         }
                //       }
                //     };
                //     result
                //   }

                // expand <head>
                let mut head = self.lower_expr(head);
                let head_sp = head.span;
                let desugared_span = self.mark_span_with_reason(
                    CompilerDesugaringKind::ForLoop,
                    head_sp,
                    None,
                );
                head.span = desugared_span;

                let iter = Ident::with_empty_ctxt(sym::iter);

                let next_ident = Ident::with_empty_ctxt(sym::__next);
                let (next_pat, next_pat_hid) = self.pat_ident_binding_mode(
                    desugared_span,
                    next_ident,
                    hir::BindingAnnotation::Mutable,
                );

                // `::std::option::Option::Some(val) => __next = val`
                let pat_arm = {
                    let val_ident = Ident::with_empty_ctxt(sym::val);
                    let (val_pat, val_pat_hid) = self.pat_ident(pat.span, val_ident);
                    let val_expr = P(self.expr_ident(pat.span, val_ident, val_pat_hid));
                    let next_expr = P(self.expr_ident(pat.span, next_ident, next_pat_hid));
                    let assign = P(self.expr(
                        pat.span,
                        hir::ExprKind::Assign(next_expr, val_expr),
                        ThinVec::new(),
                    ));
                    let some_pat = self.pat_some(pat.span, val_pat);
                    self.arm(hir_vec![some_pat], assign)
                };

                // `::std::option::Option::None => break`
                let break_arm = {
                    let break_expr =
                        self.with_loop_scope(e.id, |this| this.expr_break(e.span, ThinVec::new()));
                    let pat = self.pat_none(e.span);
                    self.arm(hir_vec![pat], break_expr)
                };

                // `mut iter`
                let (iter_pat, iter_pat_nid) = self.pat_ident_binding_mode(
                    desugared_span,
                    iter,
                    hir::BindingAnnotation::Mutable
                );

                // `match ::std::iter::Iterator::next(&mut iter) { ... }`
                let match_expr = {
                    let iter = P(self.expr_ident(head_sp, iter, iter_pat_nid));
                    let ref_mut_iter = self.expr_mut_addr_of(head_sp, iter);
                    let next_path = &[sym::iter, sym::Iterator, sym::next];
                    let next_expr = P(self.expr_call_std_path(
                        head_sp,
                        next_path,
                        hir_vec![ref_mut_iter],
                    ));
                    let arms = hir_vec![pat_arm, break_arm];

                    P(self.expr(
                        head_sp,
                        hir::ExprKind::Match(
                            next_expr,
                            arms,
                            hir::MatchSource::ForLoopDesugar
                        ),
                        ThinVec::new(),
                    ))
                };
                let match_stmt = self.stmt(head_sp, hir::StmtKind::Expr(match_expr));

                let next_expr = P(self.expr_ident(head_sp, next_ident, next_pat_hid));

                // `let mut __next`
                let next_let = self.stmt_let_pat(
                    desugared_span,
                    None,
                    next_pat,
                    hir::LocalSource::ForLoopDesugar,
                );

                // `let <pat> = __next`
                let pat = self.lower_pat(pat);
                let pat_let = self.stmt_let_pat(
                    head_sp,
                    Some(next_expr),
                    pat,
                    hir::LocalSource::ForLoopDesugar,
                );

                let body_block = self.with_loop_scope(e.id, |this| this.lower_block(body, false));
                let body_expr = P(self.expr_block(body_block, ThinVec::new()));
                let body_stmt = self.stmt(body.span, hir::StmtKind::Expr(body_expr));

                let loop_block = P(self.block_all(
                    e.span,
                    hir_vec![next_let, match_stmt, pat_let, body_stmt],
                    None,
                ));

                // `[opt_ident]: loop { ... }`
                let loop_expr = hir::ExprKind::Loop(
                    loop_block,
                    self.lower_label(opt_label),
                    hir::LoopSource::ForLoop,
                );
                let loop_expr = P(hir::Expr {
                    hir_id: self.lower_node_id(e.id),
                    node: loop_expr,
                    span: e.span,
                    attrs: ThinVec::new(),
                });

                // `mut iter => { ... }`
                let iter_arm = self.arm(hir_vec![iter_pat], loop_expr);

                // `match ::std::iter::IntoIterator::into_iter(<head>) { ... }`
                let into_iter_expr = {
                    let into_iter_path =
                        &[sym::iter, sym::IntoIterator, sym::into_iter];
                    P(self.expr_call_std_path(
                        head_sp,
                        into_iter_path,
                        hir_vec![head],
                    ))
                };

                let match_expr = P(self.expr_match(
                    head_sp,
                    into_iter_expr,
                    hir_vec![iter_arm],
                    hir::MatchSource::ForLoopDesugar,
                ));

                // This is effectively `{ let _result = ...; _result }`.
                // The construct was introduced in #21984.
                // FIXME(60253): Is this still necessary?
                // Also, add the attributes to the outer returned expr node.
                return self.expr_drop_temps(head_sp, match_expr, e.attrs.clone())
            }

            // Desugar `ExprKind::Try`
            // from: `<expr>?`
            ExprKind::Try(ref sub_expr) => {
                // into:
                //
                // match Try::into_result(<expr>) {
                //     Ok(val) => #[allow(unreachable_code)] val,
                //     Err(err) => #[allow(unreachable_code)]
                //                 // If there is an enclosing `catch {...}`
                //                 break 'catch_target Try::from_error(From::from(err)),
                //                 // Otherwise
                //                 return Try::from_error(From::from(err)),
                // }

                let unstable_span = self.mark_span_with_reason(
                    CompilerDesugaringKind::QuestionMark,
                    e.span,
                    self.allow_try_trait.clone(),
                );
                let try_span = self.sess.source_map().end_point(e.span);
                let try_span = self.mark_span_with_reason(
                    CompilerDesugaringKind::QuestionMark,
                    try_span,
                    self.allow_try_trait.clone(),
                );

                // `Try::into_result(<expr>)`
                let discr = {
                    // expand <expr>
                    let sub_expr = self.lower_expr(sub_expr);

                    let path = &[sym::ops, sym::Try, sym::into_result];
                    P(self.expr_call_std_path(
                        unstable_span,
                        path,
                        hir_vec![sub_expr],
                    ))
                };

                // `#[allow(unreachable_code)]`
                let attr = {
                    // `allow(unreachable_code)`
                    let allow = {
                        let allow_ident = Ident::with_empty_ctxt(sym::allow).with_span_pos(e.span);
                        let uc_ident = Ident::with_empty_ctxt(sym::unreachable_code)
                            .with_span_pos(e.span);
                        let uc_nested = attr::mk_nested_word_item(uc_ident);
                        attr::mk_list_item(e.span, allow_ident, vec![uc_nested])
                    };
                    attr::mk_spanned_attr_outer(e.span, attr::mk_attr_id(), allow)
                };
                let attrs = vec![attr];

                // `Ok(val) => #[allow(unreachable_code)] val,`
                let ok_arm = {
                    let val_ident = Ident::with_empty_ctxt(sym::val);
                    let (val_pat, val_pat_nid) = self.pat_ident(e.span, val_ident);
                    let val_expr = P(self.expr_ident_with_attrs(
                        e.span,
                        val_ident,
                        val_pat_nid,
                        ThinVec::from(attrs.clone()),
                    ));
                    let ok_pat = self.pat_ok(e.span, val_pat);

                    self.arm(hir_vec![ok_pat], val_expr)
                };

                // `Err(err) => #[allow(unreachable_code)]
                //              return Try::from_error(From::from(err)),`
                let err_arm = {
                    let err_ident = Ident::with_empty_ctxt(sym::err);
                    let (err_local, err_local_nid) = self.pat_ident(try_span, err_ident);
                    let from_expr = {
                        let from_path = &[sym::convert, sym::From, sym::from];
                        let err_expr = self.expr_ident(try_span, err_ident, err_local_nid);
                        self.expr_call_std_path(try_span, from_path, hir_vec![err_expr])
                    };
                    let from_err_expr =
                        self.wrap_in_try_constructor(sym::from_error, from_expr, unstable_span);
                    let thin_attrs = ThinVec::from(attrs);
                    let catch_scope = self.catch_scopes.last().map(|x| *x);
                    let ret_expr = if let Some(catch_node) = catch_scope {
                        let target_id = Ok(self.lower_node_id(catch_node));
                        P(self.expr(
                            try_span,
                            hir::ExprKind::Break(
                                hir::Destination {
                                    label: None,
                                    target_id,
                                },
                                Some(from_err_expr),
                            ),
                            thin_attrs,
                        ))
                    } else {
                        P(self.expr(try_span, hir::ExprKind::Ret(Some(from_err_expr)), thin_attrs))
                    };

                    let err_pat = self.pat_err(try_span, err_local);
                    self.arm(hir_vec![err_pat], ret_expr)
                };

                hir::ExprKind::Match(
                    discr,
                    hir_vec![err_arm, ok_arm],
                    hir::MatchSource::TryDesugar,
                )
            }

            ExprKind::Mac(_) => panic!("Shouldn't exist here"),
        };

        hir::Expr {
            hir_id: self.lower_node_id(e.id),
            node: kind,
            span: e.span,
            attrs: e.attrs.clone(),
        }
    }

    fn lower_stmt(&mut self, s: &Stmt) -> SmallVec<[hir::Stmt; 1]> {
        smallvec![match s.node {
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
            StmtKind::Expr(ref e) => {
                hir::Stmt {
                    hir_id: self.lower_node_id(s.id),
                    node: hir::StmtKind::Expr(P(self.lower_expr(e))),
                    span: s.span,
                }
            },
            StmtKind::Semi(ref e) => {
                hir::Stmt {
                    hir_id: self.lower_node_id(s.id),
                    node: hir::StmtKind::Semi(P(self.lower_expr(e))),
                    span: s.span,
                }
            },
            StmtKind::Mac(..) => panic!("Shouldn't exist here"),
        }]
    }

    fn lower_capture_clause(&mut self, c: CaptureBy) -> hir::CaptureClause {
        match c {
            CaptureBy::Value => hir::CaptureByValue,
            CaptureBy::Ref => hir::CaptureByRef,
        }
    }

    /// If an `explicit_owner` is given, this method allocates the `HirId` in
    /// the address space of that item instead of the item currently being
    /// lowered. This can happen during `lower_impl_item_ref()` where we need to
    /// lower a `Visibility` value although we haven't lowered the owning
    /// `ImplItem` in question yet.
    fn lower_visibility(
        &mut self,
        v: &Visibility,
        explicit_owner: Option<NodeId>,
    ) -> hir::Visibility {
        let node = match v.node {
            VisibilityKind::Public => hir::VisibilityKind::Public,
            VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            VisibilityKind::Restricted { ref path, id } => {
                debug!("lower_visibility: restricted path id = {:?}", id);
                let lowered_id = if let Some(owner) = explicit_owner {
                    self.lower_node_id_with_owner(id, owner)
                } else {
                    self.lower_node_id(id)
                };
                let res = self.expect_full_res(id);
                let res = self.lower_res(res);
                hir::VisibilityKind::Restricted {
                    path: P(self.lower_path_extra(
                        res,
                        path,
                        ParamMode::Explicit,
                        explicit_owner,
                    )),
                    hir_id: lowered_id,
                }
            },
            VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
        };
        respan(v.span, node)
    }

    fn lower_defaultness(&self, d: Defaultness, has_value: bool) -> hir::Defaultness {
        match d {
            Defaultness::Default => hir::Defaultness::Default {
                has_value: has_value,
            },
            Defaultness::Final => {
                assert!(has_value);
                hir::Defaultness::Final
            }
        }
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

    fn lower_impl_polarity(&mut self, i: ImplPolarity) -> hir::ImplPolarity {
        match i {
            ImplPolarity::Positive => hir::ImplPolarity::Positive,
            ImplPolarity::Negative => hir::ImplPolarity::Negative,
        }
    }

    fn lower_trait_bound_modifier(&mut self, f: TraitBoundModifier) -> hir::TraitBoundModifier {
        match f {
            TraitBoundModifier::None => hir::TraitBoundModifier::None,
            TraitBoundModifier::Maybe => hir::TraitBoundModifier::Maybe,
        }
    }

    // Helper methods for building HIR.

    fn arm(&mut self, pats: hir::HirVec<P<hir::Pat>>, expr: P<hir::Expr>) -> hir::Arm {
        hir::Arm {
            hir_id: self.next_id(),
            attrs: hir_vec![],
            pats,
            guard: None,
            span: expr.span,
            body: expr,
        }
    }

    fn field(&mut self, ident: Ident, expr: P<hir::Expr>, span: Span) -> hir::Field {
        hir::Field {
            hir_id: self.next_id(),
            ident,
            span,
            expr,
            is_shorthand: false,
        }
    }

    fn expr_break(&mut self, span: Span, attrs: ThinVec<Attribute>) -> P<hir::Expr> {
        let expr_break = hir::ExprKind::Break(self.lower_loop_destination(None), None);
        P(self.expr(span, expr_break, attrs))
    }

    fn expr_call(
        &mut self,
        span: Span,
        e: P<hir::Expr>,
        args: hir::HirVec<hir::Expr>,
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::Call(e, args), ThinVec::new())
    }

    // Note: associated functions must use `expr_call_std_path`.
    fn expr_call_std_path(
        &mut self,
        span: Span,
        path_components: &[Symbol],
        args: hir::HirVec<hir::Expr>,
    ) -> hir::Expr {
        let path = P(self.expr_std_path(span, path_components, None, ThinVec::new()));
        self.expr_call(span, path, args)
    }

    // Create an expression calling an associated function of an std type.
    //
    // Associated functions cannot be resolved through the normal `std_path` function,
    // as they are resolved differently and so cannot use `expr_call_std_path`.
    //
    // This function accepts the path component (`ty_path_components`) separately from
    // the name of the associated function (`assoc_fn_name`) in order to facilitate
    // separate resolution of the type and creation of a path referring to its associated
    // function.
    fn expr_call_std_assoc_fn(
        &mut self,
        ty_path_id: hir::HirId,
        span: Span,
        ty_path_components: &[Symbol],
        assoc_fn_name: &str,
        args: hir::HirVec<hir::Expr>,
    ) -> hir::ExprKind {
        let ty_path = P(self.std_path(span, ty_path_components, None, false));
        let ty = P(self.ty_path(ty_path_id, span, hir::QPath::Resolved(None, ty_path)));
        let fn_seg = P(hir::PathSegment::from_ident(Ident::from_str(assoc_fn_name)));
        let fn_path = hir::QPath::TypeRelative(ty, fn_seg);
        let fn_expr = P(self.expr(span, hir::ExprKind::Path(fn_path), ThinVec::new()));
        hir::ExprKind::Call(fn_expr, args)
    }

    fn expr_ident(&mut self, span: Span, ident: Ident, binding: hir::HirId) -> hir::Expr {
        self.expr_ident_with_attrs(span, ident, binding, ThinVec::new())
    }

    fn expr_ident_with_attrs(
        &mut self,
        span: Span,
        ident: Ident,
        binding: hir::HirId,
        attrs: ThinVec<Attribute>,
    ) -> hir::Expr {
        let expr_path = hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            P(hir::Path {
                span,
                res: Res::Local(binding),
                segments: hir_vec![hir::PathSegment::from_ident(ident)],
            }),
        ));

        self.expr(span, expr_path, attrs)
    }

    fn expr_mut_addr_of(&mut self, span: Span, e: P<hir::Expr>) -> hir::Expr {
        self.expr(span, hir::ExprKind::AddrOf(hir::MutMutable, e), ThinVec::new())
    }

    fn expr_std_path(
        &mut self,
        span: Span,
        components: &[Symbol],
        params: Option<P<hir::GenericArgs>>,
        attrs: ThinVec<Attribute>,
    ) -> hir::Expr {
        let path = self.std_path(span, components, params, true);
        self.expr(
            span,
            hir::ExprKind::Path(hir::QPath::Resolved(None, P(path))),
            attrs,
        )
    }

    /// Wrap the given `expr` in a terminating scope using `hir::ExprKind::DropTemps`.
    ///
    /// In terms of drop order, it has the same effect as wrapping `expr` in
    /// `{ let _t = $expr; _t }` but should provide better compile-time performance.
    ///
    /// The drop order can be important in e.g. `if expr { .. }`.
    fn expr_drop_temps(
        &mut self,
        span: Span,
        expr: P<hir::Expr>,
        attrs: ThinVec<Attribute>
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::DropTemps(expr), attrs)
    }

    fn expr_match(
        &mut self,
        span: Span,
        arg: P<hir::Expr>,
        arms: hir::HirVec<hir::Arm>,
        source: hir::MatchSource,
    ) -> hir::Expr {
        self.expr(span, hir::ExprKind::Match(arg, arms, source), ThinVec::new())
    }

    fn expr_block(&mut self, b: P<hir::Block>, attrs: ThinVec<Attribute>) -> hir::Expr {
        self.expr(b.span, hir::ExprKind::Block(b, None), attrs)
    }

    fn expr_unit(&mut self, sp: Span) -> hir::Expr {
        self.expr_tuple(sp, hir_vec![])
    }

    fn expr_tuple(&mut self, sp: Span, exprs: hir::HirVec<hir::Expr>) -> hir::Expr {
        self.expr(sp, hir::ExprKind::Tup(exprs), ThinVec::new())
    }

    fn expr(&mut self, span: Span, node: hir::ExprKind, attrs: ThinVec<Attribute>) -> hir::Expr {
        hir::Expr {
            hir_id: self.next_id(),
            node,
            span,
            attrs,
        }
    }

    fn stmt(&mut self, span: Span, node: hir::StmtKind) -> hir::Stmt {
        hir::Stmt { span, node, hir_id: self.next_id() }
    }

    fn stmt_let_pat(
        &mut self,
        span: Span,
        init: Option<P<hir::Expr>>,
        pat: P<hir::Pat>,
        source: hir::LocalSource,
    ) -> hir::Stmt {
        let local = hir::Local {
            pat,
            ty: None,
            init,
            hir_id: self.next_id(),
            span,
            source,
            attrs: ThinVec::new()
        };
        self.stmt(span, hir::StmtKind::Local(P(local)))
    }

    fn expr_block_empty(&mut self, span: Span) -> hir::Expr {
        let blk = self.block_all(span, hir_vec![], None);
        self.expr_block(P(blk), ThinVec::new())
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

    fn expr_unsafe(&mut self, expr: P<hir::Expr>) -> hir::Expr {
        let hir_id = self.next_id();
        let span = expr.span;
        self.expr(
            span,
            hir::ExprKind::Block(P(hir::Block {
                stmts: hir_vec![],
                expr: Some(expr),
                hir_id,
                rules: hir::UnsafeBlock(hir::CompilerGenerated),
                span,
                targeted_by_break: false,
            }), None),
            ThinVec::new(),
        )
    }

    /// Constructs a `true` or `false` literal expression.
    fn expr_bool(&mut self, span: Span, val: bool) -> hir::Expr {
        let lit = Spanned { span, node: LitKind::Bool(val) };
        self.expr(span, hir::ExprKind::Lit(lit), ThinVec::new())
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
        let (path, res) = self.resolver
            .resolve_str_path(span, self.crate_root, components, is_value);

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
            res: res.map_id(|_| panic!("unexpected node_id")),
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

            AnonymousLifetimeMode::Replace(replacement) => {
                self.new_replacement_lifetime(replacement, span)
            }
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

            AnonymousLifetimeMode::Replace(replacement) => {
                self.new_replacement_lifetime(replacement, span)
            }

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

            // We don't need to do any replacement here as this lifetime
            // doesn't refer to an elided lifetime elsewhere in the function
            // signature.
            AnonymousLifetimeMode::Replace(_) => {}
        }

        self.new_implicit_lifetime(span)
    }

    fn new_replacement_lifetime(
        &mut self,
        replacement: LtReplacement,
        span: Span,
    ) -> hir::Lifetime {
        let hir_id = self.next_id();
        self.replace_elided_lifetime(hir_id, span, replacement)
    }

    fn new_implicit_lifetime(&mut self, span: Span) -> hir::Lifetime {
        hir::Lifetime {
            hir_id: self.next_id(),
            span,
            name: hir::LifetimeName::Implicit,
        }
    }

    fn maybe_lint_bare_trait(&self, span: Span, id: NodeId, is_global: bool) {
        self.sess.buffer_lint_with_diagnostic(
            builtin::BARE_TRAIT_OBJECTS,
            id,
            span,
            "trait objects without an explicit `dyn` are deprecated",
            builtin::BuiltinLintDiagnostics::BareTraitObject(span, is_global),
        )
    }

    fn wrap_in_try_constructor(
        &mut self,
        method: Symbol,
        e: hir::Expr,
        unstable_span: Span,
    ) -> P<hir::Expr> {
        let path = &[sym::ops, sym::Try, method];
        let from_err = P(self.expr_std_path(unstable_span, path, None,
                                            ThinVec::new()));
        P(self.expr_call(e.span, from_err, hir_vec![e]))
    }

    fn lower_await(
        &mut self,
        await_span: Span,
        expr: &ast::Expr,
    ) -> hir::ExprKind {
        // to:
        //
        // {
        //     let mut pinned = <expr>;
        //     loop {
        //         match ::std::future::poll_with_tls_context(unsafe {
        //             ::std::pin::Pin::new_unchecked(&mut pinned)
        //         }) {
        //             ::std::task::Poll::Ready(result) => break result,
        //             ::std::task::Poll::Pending => {},
        //         }
        //         yield ();
        //     }
        // }
        match self.generator_kind {
            Some(hir::GeneratorKind::Async) => {},
            Some(hir::GeneratorKind::Gen) |
            None => {
                let mut err = struct_span_err!(
                    self.sess,
                    await_span,
                    E0728,
                    "`await` is only allowed inside `async` functions and blocks"
                );
                err.span_label(await_span, "only allowed inside `async` functions and blocks");
                if let Some(item_sp) = self.current_item {
                    err.span_label(item_sp, "this is not `async`");
                }
                err.emit();
            }
        }
        let span = self.mark_span_with_reason(
            CompilerDesugaringKind::Await,
            await_span,
            None,
        );
        let gen_future_span = self.mark_span_with_reason(
            CompilerDesugaringKind::Await,
            await_span,
            self.allow_gen_future.clone(),
        );

        // let mut pinned = <expr>;
        let expr = P(self.lower_expr(expr));
        let pinned_ident = Ident::with_empty_ctxt(sym::pinned);
        let (pinned_pat, pinned_pat_hid) = self.pat_ident_binding_mode(
            span,
            pinned_ident,
            hir::BindingAnnotation::Mutable,
        );
        let pinned_let = self.stmt_let_pat(
            span,
            Some(expr),
            pinned_pat,
            hir::LocalSource::AwaitDesugar,
        );

        // ::std::future::poll_with_tls_context(unsafe {
        //     ::std::pin::Pin::new_unchecked(&mut pinned)
        // })`
        let poll_expr = {
            let pinned = P(self.expr_ident(span, pinned_ident, pinned_pat_hid));
            let ref_mut_pinned = self.expr_mut_addr_of(span, pinned);
            let pin_ty_id = self.next_id();
            let new_unchecked_expr_kind = self.expr_call_std_assoc_fn(
                pin_ty_id,
                span,
                &[sym::pin, sym::Pin],
                "new_unchecked",
                hir_vec![ref_mut_pinned],
            );
            let new_unchecked = P(self.expr(span, new_unchecked_expr_kind, ThinVec::new()));
            let unsafe_expr = self.expr_unsafe(new_unchecked);
            P(self.expr_call_std_path(
                gen_future_span,
                &[sym::future, sym::poll_with_tls_context],
                hir_vec![unsafe_expr],
            ))
        };

        // `::std::task::Poll::Ready(result) => break result`
        let loop_node_id = self.sess.next_node_id();
        let loop_hir_id = self.lower_node_id(loop_node_id);
        let ready_arm = {
            let x_ident = Ident::with_empty_ctxt(sym::result);
            let (x_pat, x_pat_hid) = self.pat_ident(span, x_ident);
            let x_expr = P(self.expr_ident(span, x_ident, x_pat_hid));
            let ready_pat = self.pat_std_enum(
                span,
                &[sym::task, sym::Poll, sym::Ready],
                hir_vec![x_pat],
            );
            let break_x = self.with_loop_scope(loop_node_id, |this| {
                let expr_break = hir::ExprKind::Break(
                    this.lower_loop_destination(None),
                    Some(x_expr),
                );
                P(this.expr(await_span, expr_break, ThinVec::new()))
            });
            self.arm(hir_vec![ready_pat], break_x)
        };

        // `::std::task::Poll::Pending => {}`
        let pending_arm = {
            let pending_pat = self.pat_std_enum(
                span,
                &[sym::task, sym::Poll, sym::Pending],
                hir_vec![],
            );
            let empty_block = P(self.expr_block_empty(span));
            self.arm(hir_vec![pending_pat], empty_block)
        };

        let match_stmt = {
            let match_expr = P(self.expr_match(
                span,
                poll_expr,
                hir_vec![ready_arm, pending_arm],
                hir::MatchSource::AwaitDesugar,
            ));
            self.stmt(span, hir::StmtKind::Expr(match_expr))
        };

        let yield_stmt = {
            let unit = self.expr_unit(span);
            let yield_expr = P(self.expr(
                span,
                hir::ExprKind::Yield(P(unit), hir::YieldSource::Await),
                ThinVec::new(),
            ));
            self.stmt(span, hir::StmtKind::Expr(yield_expr))
        };

        let loop_block = P(self.block_all(
            span,
            hir_vec![match_stmt, yield_stmt],
            None,
        ));

        let loop_expr = P(hir::Expr {
            hir_id: loop_hir_id,
            node: hir::ExprKind::Loop(
                loop_block,
                None,
                hir::LoopSource::Loop,
            ),
            span,
            attrs: ThinVec::new(),
        });

        hir::ExprKind::Block(
            P(self.block_all(span, hir_vec![pinned_let], Some(loop_expr))),
            None,
        )
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
