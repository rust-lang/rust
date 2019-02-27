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
//! ID from an AST node in a single HIR node (you can assume that AST node IDs
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
use crate::hir::def_id::{DefId, DefIndex, DefIndexAddressSpace, CRATE_DEF_INDEX};
use crate::hir::def::{Def, PathResolution, PerNS};
use crate::hir::{GenericArg, ConstArg};
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
use std::fmt::Debug;
use std::mem;
use smallvec::SmallVec;
use syntax::attr;
use syntax::ast;
use syntax::ast::*;
use syntax::errors;
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::source_map::{self, respan, CompilerDesugaringKind, Spanned};
use syntax::std_inject;
use syntax::symbol::{keywords, Symbol};
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax::parse::token::Token;
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, MultiSpan};

const HIR_ID_COUNTER_LOCKED: u32 = 0xFFFFFFFF;

pub struct LoweringContext<'a> {
    crate_root: Option<&'static str>,

    // Used to assign ids to HIR nodes that do not directly correspond to an AST node.
    sess: &'a Session,

    cstore: &'a dyn CrateStore,

    resolver: &'a mut dyn Resolver,

    /// The items being lowered are collected here.
    items: BTreeMap<NodeId, hir::Item>,

    trait_items: BTreeMap<hir::TraitItemId, hir::TraitItem>,
    impl_items: BTreeMap<hir::ImplItemId, hir::ImplItem>,
    bodies: BTreeMap<hir::BodyId, hir::Body>,
    exported_macros: Vec<hir::MacroDef>,

    trait_impls: BTreeMap<DefId, Vec<NodeId>>,
    trait_auto_impl: BTreeMap<DefId, NodeId>,

    modules: BTreeMap<NodeId, hir::ModuleItems>,

    is_generator: bool,

    catch_scopes: Vec<NodeId>,
    loop_scopes: Vec<NodeId>,
    is_in_loop_condition: bool,
    is_in_trait_impl: bool,

    /// What to do when we encounter either an "anonymous lifetime
    /// reference". The term "anonymous" is meant to encompass both
    /// `'_` lifetimes as well as fully elided cases where nothing is
    /// written at all (e.g., `&T` or `std::cell::Ref<T>`).
    anonymous_lifetime_mode: AnonymousLifetimeMode,

    // Used to create lifetime definitions from in-band lifetime usages.
    // e.g., `fn foo(x: &'x u8) -> &'x u8` to `fn foo<'x>(x: &'x u8) -> &'x u8`
    // When a named lifetime is encountered in a function or impl header and
    // has not been defined
    // (i.e., it doesn't appear in the in_scope_lifetimes list), it is added
    // to this list. The results of this list are then added to the list of
    // lifetime definitions in the corresponding impl or function generics.
    lifetimes_to_define: Vec<(Span, ParamName)>,

    // Whether or not in-band lifetimes are being collected. This is used to
    // indicate whether or not we're in a place where new lifetimes will result
    // in in-band lifetime definitions, such a function or an impl header,
    // including implicit lifetimes from `impl_header_lifetime_elision`.
    is_collecting_in_band_lifetimes: bool,

    // Currently in-scope lifetimes defined in impl headers, fn headers, or HRTB.
    // When `is_collectin_in_band_lifetimes` is true, each lifetime is checked
    // against this list to see if it is already in-scope, or if a definition
    // needs to be created for it.
    in_scope_lifetimes: Vec<Ident>,

    current_module: NodeId,

    type_def_lifetime_params: DefIdMap<usize>,

    current_hir_id_owner: Vec<(DefIndex, u32)>,
    item_local_id_counters: NodeMap<u32>,
    node_id_to_hir_id: IndexVec<NodeId, hir::HirId>,
}

pub trait Resolver {
    /// Resolve a path generated by the lowerer when expanding `for`, `if let`, etc.
    fn resolve_hir_path(
        &mut self,
        path: &ast::Path,
        is_value: bool,
    ) -> hir::Path;

    /// Obtain the resolution for a `NodeId`.
    fn get_resolution(&mut self, id: NodeId) -> Option<PathResolution>;

    /// Obtain the possible resolutions for the given `use` statement.
    fn get_import(&mut self, id: NodeId) -> PerNS<Option<PathResolution>>;

    /// We must keep the set of definitions up to date as we add nodes that weren't in the AST.
    /// This should only return `None` during testing.
    fn definitions(&mut self) -> &mut Definitions;

    /// Given suffix `["b", "c", "d"]`, creates a HIR path for `[::crate_root]::b::c::d` and
    /// resolves it based on `is_value`.
    fn resolve_str_path(
        &mut self,
        span: Span,
        crate_root: Option<&str>,
        components: &[&str],
        is_value: bool,
    ) -> hir::Path;
}

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
    /// information later. It is `None` when no information about the context should be stored,
    /// e.g., for consts and statics.
    Existential(Option<DefId>),

    /// `impl Trait` is not accepted in this position.
    Disallowed(ImplTraitPosition),
}

/// Position in which `impl Trait` is disallowed. Used for error reporting.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitPosition {
    Binding,
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
            Existential(did) => Existential(*did),
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
        crate_root: std_inject::injected_crate_name(),
        sess,
        cstore,
        resolver,
        items: BTreeMap::new(),
        trait_items: BTreeMap::new(),
        impl_items: BTreeMap::new(),
        bodies: BTreeMap::new(),
        trait_impls: BTreeMap::new(),
        trait_auto_impl: BTreeMap::new(),
        modules: BTreeMap::new(),
        exported_macros: Vec::new(),
        catch_scopes: Vec::new(),
        loop_scopes: Vec::new(),
        is_in_loop_condition: false,
        anonymous_lifetime_mode: AnonymousLifetimeMode::PassThrough,
        type_def_lifetime_params: Default::default(),
        current_module: CRATE_NODE_ID,
        current_hir_id_owner: vec![(CRATE_DEF_INDEX, 0)],
        item_local_id_counters: Default::default(),
        node_id_to_hir_id: IndexVec::new(),
        is_generator: false,
        is_in_trait_impl: false,
        lifetimes_to_define: Vec::new(),
        is_collecting_in_band_lifetimes: false,
        in_scope_lifetimes: Vec::new(),
    }.lower_crate(krate)
}

#[derive(Copy, Clone, PartialEq)]
enum ParamMode {
    /// Any path in a type context.
    Explicit,
    /// The `module::Type` in `module::Type::method` in an expression.
    Optional,
}

#[derive(Debug)]
struct LoweredNodeId {
    node_id: NodeId,
    hir_id: hir::HirId,
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
}

struct ImplTraitTypeIdVisitor<'a> { ids: &'a mut SmallVec<[hir::ItemId; 1]> }

impl<'a, 'b> Visitor<'a> for ImplTraitTypeIdVisitor<'b> {
    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            | TyKind::Typeof(_)
            | TyKind::BareFn(_)
            => return,

            TyKind::ImplTrait(id, _) => self.ids.push(hir::ItemId { id }),
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
        struct MiscCollector<'lcx, 'interner: 'lcx> {
            lctx: &'lcx mut LoweringContext<'interner>,
        }

        impl<'lcx, 'interner> Visitor<'lcx> for MiscCollector<'lcx, 'interner> {
            fn visit_item(&mut self, item: &'lcx Item) {
                self.lctx.allocate_hir_id_counter(item.id, item);

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
                    _ => {}
                }
                visit::walk_item(self, item);
            }

            fn visit_trait_item(&mut self, item: &'lcx TraitItem) {
                self.lctx.allocate_hir_id_counter(item.id, item);
                visit::walk_trait_item(self, item);
            }

            fn visit_impl_item(&mut self, item: &'lcx ImplItem) {
                self.lctx.allocate_hir_id_counter(item.id, item);
                visit::walk_impl_item(self, item);
            }
        }

        struct ItemLowerer<'lcx, 'interner: 'lcx> {
            lctx: &'lcx mut LoweringContext<'interner>,
        }

        impl<'lcx, 'interner> ItemLowerer<'lcx, 'interner> {
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

        impl<'lcx, 'interner> Visitor<'lcx> for ItemLowerer<'lcx, 'interner> {
            fn visit_mod(&mut self, m: &'lcx Mod, _s: Span, _attrs: &[Attribute], n: NodeId) {
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

            fn visit_item(&mut self, item: &'lcx Item) {
                let mut item_lowered = true;
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    if let Some(hir_item) = lctx.lower_item(item) {
                        lctx.insert_item(item.id, hir_item);
                    } else {
                        item_lowered = false;
                    }
                });

                if item_lowered {
                    let item_generics = match self.lctx.items.get(&item.id).unwrap().node {
                        hir::ItemKind::Impl(_, _, _, ref generics, ..)
                        | hir::ItemKind::Trait(_, _, ref generics, ..) => {
                            generics.params.clone()
                        }
                        _ => HirVec::new(),
                    };

                    self.lctx.with_parent_impl_lifetime_defs(&item_generics, |this| {
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

            fn visit_trait_item(&mut self, item: &'lcx TraitItem) {
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    let id = hir::TraitItemId { node_id: item.id };
                    let hir_item = lctx.lower_trait_item(item);
                    lctx.trait_items.insert(id, hir_item);
                    lctx.modules.get_mut(&lctx.current_module).unwrap().trait_items.insert(id);
                });

                visit::walk_trait_item(self, item);
            }

            fn visit_impl_item(&mut self, item: &'lcx ImplItem) {
                self.lctx.with_hir_id_owner(item.id, |lctx| {
                    let id = hir::ImplItemId { node_id: item.id };
                    let hir_item = lctx.lower_impl_item(item);
                    lctx.impl_items.insert(id, hir_item);
                    lctx.modules.get_mut(&lctx.current_module).unwrap().impl_items.insert(id);
                });
                visit::walk_impl_item(self, item);
            }
        }

        self.lower_node_id(CRATE_NODE_ID);
        debug_assert!(self.node_id_to_hir_id[CRATE_NODE_ID] == hir::CRATE_HIR_ID);

        visit::walk_crate(&mut MiscCollector { lctx: &mut self }, c);
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
            trait_auto_impl: self.trait_auto_impl,
            modules: self.modules,
        }
    }

    fn insert_item(&mut self, id: NodeId, item: hir::Item) {
        self.items.insert(id, item);
        self.modules.get_mut(&self.current_module).unwrap().items.insert(id);
    }

    fn allocate_hir_id_counter<T: Debug>(&mut self, owner: NodeId, debug: &T) -> LoweredNodeId {
        if self.item_local_id_counters.insert(owner, 0).is_some() {
            bug!(
                "Tried to allocate item_local_id_counter for {:?} twice",
                debug
            );
        }
        // Always allocate the first `HirId` for the owner itself.
        self.lower_node_id_with_owner(owner, owner)
    }

    fn lower_node_id_generic<F>(&mut self, ast_node_id: NodeId, alloc_hir_id: F) -> LoweredNodeId
    where
        F: FnOnce(&mut Self) -> hir::HirId,
    {
        if ast_node_id == DUMMY_NODE_ID {
            return LoweredNodeId {
                node_id: DUMMY_NODE_ID,
                hir_id: hir::DUMMY_HIR_ID,
            };
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
            LoweredNodeId {
                node_id: ast_node_id,
                hir_id,
            }
        } else {
            LoweredNodeId {
                node_id: ast_node_id,
                hir_id: existing_hir_id,
            }
        }
    }

    fn with_hir_id_owner<F, T>(&mut self, owner: NodeId, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let counter = self.item_local_id_counters
            .insert(owner, HIR_ID_COUNTER_LOCKED)
            .unwrap_or_else(|| panic!("No item_local_id_counters entry for {:?}", owner));
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
    fn lower_node_id(&mut self, ast_node_id: NodeId) -> LoweredNodeId {
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

    fn lower_node_id_with_owner(&mut self, ast_node_id: NodeId, owner: NodeId) -> LoweredNodeId {
        self.lower_node_id_generic(ast_node_id, |this| {
            let local_id_counter = this
                .item_local_id_counters
                .get_mut(&owner)
                .expect("called lower_node_id_with_owner before allocate_hir_id_counter");
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
                .expect("You forgot to call `create_def_with_parent` or are lowering node ids \
                         that do not belong to the current owner");

            hir::HirId {
                owner: def_index,
                local_id: hir::ItemLocalId::from_u32(local_id),
            }
        })
    }

    fn record_body(&mut self, value: hir::Expr, decl: Option<&FnDecl>) -> hir::BodyId {
        let body = hir::Body {
            arguments: decl.map_or(hir_vec![], |decl| {
                decl.inputs.iter().map(|x| self.lower_arg(x)).collect()
            }),
            is_generator: self.is_generator,
            value,
        };
        let id = body.id();
        self.bodies.insert(id, body);
        id
    }

    fn next_id(&mut self) -> LoweredNodeId {
        self.lower_node_id(self.sess.next_node_id())
    }

    fn expect_full_def(&mut self, id: NodeId) -> Def {
        self.resolver.get_resolution(id).map_or(Def::Err, |pr| {
            if pr.unresolved_segments() != 0 {
                bug!("path not fully resolved: {:?}", pr);
            }
            pr.base_def()
        })
    }

    fn expect_full_def_from_use(&mut self, id: NodeId) -> impl Iterator<Item = Def> {
        self.resolver.get_import(id).present_items().map(|pr| {
            if pr.unresolved_segments() != 0 {
                bug!("path not fully resolved: {:?}", pr);
            }
            pr.base_def()
        })
    }

    fn diagnostic(&self) -> &errors::Handler {
        self.sess.diagnostic()
    }

    fn str_to_ident(&self, s: &'static str) -> Ident {
        Ident::with_empty_ctxt(Symbol::gensym(s))
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
        mark.set_expn_info(source_map::ExpnInfo {
            call_site: span,
            def_site: Some(span),
            format: source_map::CompilerDesugaring(reason),
            allow_internal_unstable,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition: source_map::hygiene::default_edition(),
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

    /// Creates a new hir::GenericParam for every new lifetime and
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
            .map(|(span, hir_name)| {
                let LoweredNodeId { node_id, hir_id } = self.next_id();

                // Get the name we'll use to make the def-path. Note
                // that collisions are ok here and this shouldn't
                // really show up for end-user.
                let (str_name, kind) = match hir_name {
                    ParamName::Plain(ident) => (
                        ident.as_interned_str(),
                        hir::LifetimeParamKind::InBand,
                    ),
                    ParamName::Fresh(_) => (
                        keywords::UnderscoreLifetime.name().as_interned_str(),
                        hir::LifetimeParamKind::Elided,
                    ),
                    ParamName::Error => (
                        keywords::UnderscoreLifetime.name().as_interned_str(),
                        hir::LifetimeParamKind::Error,
                    ),
                };

                // Add a definition for the in-band lifetime def.
                self.resolver.definitions().create_def_with_parent(
                    parent_id.index,
                    node_id,
                    DefPathData::LifetimeParam(str_name),
                    DefIndexAddressSpace::High,
                    Mark::root(),
                    span,
                );

                hir::GenericParam {
                    hir_id,
                    name: hir_name,
                    attrs: hir_vec![],
                    bounds: hir_vec![],
                    span,
                    pure_wrt_drop: false,
                    kind: hir::GenericParamKind::Lifetime { kind }
                }
            })
            .chain(in_band_ty_params.into_iter())
            .collect();

        (params, res)
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
    fn with_parent_impl_lifetime_defs<T, F>(&mut self,
        params: &HirVec<hir::GenericParam>,
        f: F
    ) -> T where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let old_len = self.in_scope_lifetimes.len();
        let lt_def_names = params.iter().filter_map(|param| match param.kind {
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
                    let generics = this.lower_generics(
                        generics,
                        ImplTraitContext::Universal(&mut params),
                    );
                    let res = f(this, &mut params);
                    (params, (generics, res))
                })
            },
        );

        lowered_generics.params = lowered_generics
            .params
            .iter()
            .cloned()
            .chain(in_band_defs)
            .collect();

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
        ret_ty: Option<&Ty>,
        body: impl FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    ) -> hir::ExprKind {
        let prev_is_generator = mem::replace(&mut self.is_generator, true);
        let body_expr = body(self);
        let span = body_expr.span;
        let output = match ret_ty {
            Some(ty) => FunctionRetTy::Ty(P(ty.clone())),
            None => FunctionRetTy::Default(span),
        };
        let decl = FnDecl {
            inputs: vec![],
            output,
            variadic: false
        };
        let body_id = self.record_body(body_expr, Some(&decl));
        self.is_generator = prev_is_generator;

        let capture_clause = self.lower_capture_clause(capture_clause);
        let closure_hir_id = self.lower_node_id(closure_node_id).hir_id;
        let decl = self.lower_fn_decl(&decl, None, /* impl trait allowed */ false, None);
        let generator = hir::Expr {
            hir_id: closure_hir_id,
            node: hir::ExprKind::Closure(capture_clause, decl, body_id, span,
                Some(hir::GeneratorMovability::Static)),
            span,
            attrs: ThinVec::new(),
        };

        let unstable_span = self.mark_span_with_reason(
            CompilerDesugaringKind::Async,
            span,
            Some(vec![
                Symbol::intern("gen_future"),
            ].into()),
        );
        let gen_future = self.expr_std_path(
            unstable_span, &["future", "from_generator"], None, ThinVec::new());
        hir::ExprKind::Call(P(gen_future), hir_vec![generator])
    }

    fn lower_body<F>(&mut self, decl: Option<&FnDecl>, f: F) -> hir::BodyId
    where
        F: FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    {
        let prev = mem::replace(&mut self.is_generator, false);
        let result = f(self);
        let r = self.record_body(result, decl);
        self.is_generator = prev;
        return r;
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
            "Loop scopes should be added and removed in stack order"
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

    fn with_new_scopes<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut LoweringContext<'_>) -> T,
    {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let catch_scopes = mem::replace(&mut self.catch_scopes, Vec::new());
        let loop_scopes = mem::replace(&mut self.loop_scopes, Vec::new());
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
                if let Def::Label(loop_id) = self.expect_full_def(id) {
                    Ok(self.lower_node_id(loop_id).node_id)
                } else {
                    Err(hir::LoopIdError::UnresolvedLabel)
                }
            }
            None => {
                self.loop_scopes
                    .last()
                    .cloned()
                    .map(|id| Ok(self.lower_node_id(id).node_id))
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
            TokenTree::Token(span, token) => self.lower_token(token, span),
            TokenTree::Delimited(span, delim, tts) => TokenTree::Delimited(
                span,
                delim,
                self.lower_token_stream(tts),
            ).into(),
        }
    }

    fn lower_token(&mut self, token: Token, span: Span) -> TokenStream {
        match token {
            Token::Interpolated(nt) => {
                let tts = nt.to_tokenstream(&self.sess.parse_sess, span);
                self.lower_token_stream(tts)
            }
            other => TokenTree::Token(span, other).into(),
        }
    }

    fn lower_arm(&mut self, arm: &Arm) -> hir::Arm {
        hir::Arm {
            attrs: self.lower_attrs(&arm.attrs),
            pats: arm.pats.iter().map(|x| self.lower_pat(x)).collect(),
            guard: match arm.guard {
                Some(Guard::If(ref x)) => Some(hir::Guard::If(P(self.lower_expr(x)))),
                _ => None,
            },
            body: P(self.lower_expr(&arm.body)),
        }
    }

    fn lower_ty_binding(&mut self, b: &TypeBinding,
                        itctx: ImplTraitContext<'_>) -> hir::TypeBinding {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(b.id);

        hir::TypeBinding {
            id: node_id,
            hir_id,
            ident: b.ident,
            ty: self.lower_ty(&b.ty, itctx),
            span: b.span,
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
                let id = self.lower_node_id(t.id);
                let qpath = self.lower_qpath(t.id, qself, path, ParamMode::Explicit, itctx);
                let ty = self.ty_path(id, t.span, qpath);
                if let hir::TyKind::TraitObject(..) = ty.node {
                    self.maybe_lint_bare_trait(t.span, t.id, qself.is_none() && path.is_global());
                }
                return ty;
            }
            TyKind::ImplicitSelf => hir::TyKind::Path(hir::QPath::Resolved(
                None,
                P(hir::Path {
                    def: self.expect_full_def(t.id),
                    segments: hir_vec![hir::PathSegment::from_ident(keywords::SelfUpper.ident())],
                    span: t.span,
                }),
            )),
            TyKind::Array(ref ty, ref length) => {
                hir::TyKind::Array(self.lower_ty(ty, itctx), self.lower_anon_const(length))
            }
            TyKind::Typeof(ref expr) => {
                hir::TyKind::Typeof(self.lower_anon_const(expr))
            }
            TyKind::TraitObject(ref bounds, kind) => {
                let mut lifetime_bound = None;
                let bounds = bounds
                    .iter()
                    .filter_map(|bound| match *bound {
                        GenericBound::Trait(ref ty, TraitBoundModifier::None) => {
                            Some(self.lower_poly_trait_ref(ty, itctx.reborrow()))
                        }
                        GenericBound::Trait(_, TraitBoundModifier::Maybe) => None,
                        GenericBound::Outlives(ref lifetime) => {
                            if lifetime_bound.is_none() {
                                lifetime_bound = Some(self.lower_lifetime(lifetime));
                            }
                            None
                        }
                    })
                    .collect();
                let lifetime_bound =
                    lifetime_bound.unwrap_or_else(|| self.elided_dyn_bound(t.span));
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
                        let LoweredNodeId { node_id: _, hir_id } =  self.lower_node_id(def_node_id);
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
                            hir_id,
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
                                def: Def::TyParam(DefId::local(def_index)),
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
            TyKind::Mac(_) => panic!("TyMac should have been expanded by now."),
        };

        let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(t.id);
        hir::Ty {
            node: kind,
            span: t.span,
            hir_id,
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
        // Not tracking it makes lints in rustc and clippy very fragile as
        // frequently opened issues show.
        let exist_ty_span = self.mark_span_with_reason(
            CompilerDesugaringKind::ExistentialReturnType,
            span,
            None,
        );

        let exist_ty_def_index = self
            .resolver
            .definitions()
            .opt_def_index(exist_ty_node_id)
            .unwrap();

        self.allocate_hir_id_counter(exist_ty_node_id, &"existential impl trait");

        let hir_bounds = self.with_hir_id_owner(exist_ty_node_id, lower_bounds);

        let (lifetimes, lifetime_defs) = self.lifetimes_from_impl_trait_bounds(
            exist_ty_node_id,
            exist_ty_def_index,
            &hir_bounds,
        );

        self.with_hir_id_owner(exist_ty_node_id, |lctx| {
            let LoweredNodeId { node_id: _, hir_id } = lctx.next_id();
            let exist_ty_item_kind = hir::ItemKind::Existential(hir::ExistTy {
                generics: hir::Generics {
                    params: lifetime_defs,
                    where_clause: hir::WhereClause {
                        hir_id,
                        predicates: Vec::new().into(),
                    },
                    span,
                },
                bounds: hir_bounds,
                impl_trait_fn: fn_def_id,
            });
            let exist_ty_id = lctx.lower_node_id(exist_ty_node_id);
            // Generate an `existential type Foo: Trait;` declaration.
            trace!("creating existential type with id {:#?}", exist_ty_id);

            trace!("exist ty def index: {:#?}", exist_ty_def_index);
            let exist_ty_item = hir::Item {
                id: exist_ty_id.node_id,
                hir_id: exist_ty_id.hir_id,
                ident: keywords::Invalid.ident(),
                attrs: Default::default(),
                node: exist_ty_item_kind,
                vis: respan(span.shrink_to_lo(), hir::VisibilityKind::Inherited),
                span: exist_ty_span,
            };

            // Insert the item into the global list. This usually happens
            // automatically for all AST items. But this existential type item
            // does not actually exist in the AST.
            lctx.insert_item(exist_ty_id.node_id, exist_ty_item);

            // `impl Trait` now just becomes `Foo<'a, 'b, ..>`.
            hir::TyKind::Def(hir::ItemId { id: exist_ty_id.node_id }, lifetimes)
        })
    }

    fn lifetimes_from_impl_trait_bounds(
        &mut self,
        exist_ty_id: NodeId,
        parent_index: DefIndex,
        bounds: &hir::GenericBounds,
    ) -> (HirVec<hir::GenericArg>, HirVec<hir::GenericParam>) {
        // This visitor walks over impl trait bounds and creates defs for all lifetimes which
        // appear in the bounds, excluding lifetimes that are created within the bounds.
        // E.g., `'a`, `'b`, but not `'c` in `impl for<'c> SomeTrait<'a, 'b, 'c>`.
        struct ImplTraitLifetimeCollector<'r, 'a: 'r> {
            context: &'r mut LoweringContext<'a>,
            parent: DefIndex,
            exist_ty_id: NodeId,
            collect_elided_lifetimes: bool,
            currently_bound_lifetimes: Vec<hir::LifetimeName>,
            already_defined_lifetimes: FxHashSet<hir::LifetimeName>,
            output_lifetimes: Vec<hir::GenericArg>,
            output_lifetime_params: Vec<hir::GenericParam>,
        }

        impl<'r, 'a: 'r, 'v> hir::intravisit::Visitor<'v> for ImplTraitLifetimeCollector<'r, 'a> {
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

                    let LoweredNodeId { node_id: _, hir_id } = self.context.next_id();
                    self.output_lifetimes.push(hir::GenericArg::Lifetime(hir::Lifetime {
                        hir_id,
                        span: lifetime.span,
                        name,
                    }));

                    // We need to manually create the ids here, because the
                    // definitions will go into the explicit `existential type`
                    // declaration and thus need to have their owner set to that item
                    let def_node_id = self.context.sess.next_node_id();
                    let LoweredNodeId { node_id: _, hir_id } =
                        self.context.lower_node_id_with_owner(def_node_id, self.exist_ty_id);
                    self.context.resolver.definitions().create_def_with_parent(
                        self.parent,
                        def_node_id,
                        DefPathData::LifetimeParam(name.ident().as_interned_str()),
                        DefIndexAddressSpace::High,
                        Mark::root(),
                        lifetime.span,
                    );

                    let (name, kind) = match name {
                        hir::LifetimeName::Underscore => (
                            hir::ParamName::Plain(keywords::UnderscoreLifetime.ident()),
                            hir::LifetimeParamKind::Elided,
                        ),
                        hir::LifetimeName::Param(param_name) => (
                            param_name,
                            hir::LifetimeParamKind::Explicit,
                        ),
                        _ => bug!("expected LifetimeName::Param or ParamName::Plain"),
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

        let resolution = self.resolver
            .get_resolution(id)
            .unwrap_or_else(|| PathResolution::new(Def::Err));

        let proj_start = p.segments.len() - resolution.unresolved_segments();
        let path = P(hir::Path {
            def: resolution.base_def(),
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
                    let type_def_id = match resolution.base_def() {
                        Def::AssociatedTy(def_id) if i + 2 == proj_start => {
                            Some(parent_def_id(self, def_id))
                        }
                        Def::Variant(def_id) if i + 1 == proj_start => {
                            Some(parent_def_id(self, def_id))
                        }
                        Def::Struct(def_id)
                        | Def::Union(def_id)
                        | Def::Enum(def_id)
                        | Def::TyAlias(def_id)
                        | Def::Trait(def_id) if i + 1 == proj_start =>
                        {
                            Some(def_id)
                        }
                        _ => None,
                    };
                    let parenthesized_generic_args = match resolution.base_def() {
                        // `a::b::Trait(Args)`
                        Def::Trait(..) if i + 1 == proj_start => ParenthesizedGenericArgs::Ok,
                        // `a::b::Trait(Args)::TraitItem`
                        Def::Method(..) | Def::AssociatedConst(..) | Def::AssociatedTy(..)
                            if i + 2 == proj_start =>
                        {
                            ParenthesizedGenericArgs::Ok
                        }
                        // Avoid duplicated errors.
                        Def::Err => ParenthesizedGenericArgs::Ok,
                        // An error
                        Def::Struct(..)
                        | Def::Enum(..)
                        | Def::Union(..)
                        | Def::TyAlias(..)
                        | Def::Variant(..) if i + 1 == proj_start =>
                        {
                            ParenthesizedGenericArgs::Err
                        }
                        // A warning for now, for compatibility reasons
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
        if resolution.unresolved_segments() == 0 {
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
        def: Def,
        p: &Path,
        param_mode: ParamMode,
        explicit_owner: Option<NodeId>,
    ) -> hir::Path {
        hir::Path {
            def,
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
        let def = self.expect_full_def(id);
        self.lower_path_extra(def, p, param_mode, None)
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
        let (mut generic_args, infer_types) = if let Some(ref generic_args) = segment.args {
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
                        (self.lower_angle_bracketed_parameter_data(
                            &data.as_angle_bracketed_args(),
                            param_mode,
                            itctx).0,
                         false)
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
                let no_ty_args = generic_args.args.len() == expected_lifetimes;
                let no_bindings = generic_args.bindings.is_empty();
                let (incl_angl_brckt, insertion_span, suggestion) = if no_ty_args && no_bindings {
                    // If there are no (non-implicit) generic args or associated-type
                    // bindings, our suggestion includes the angle brackets.
                    (true, path_span.shrink_to_hi(), format!("<{}>", anon_lt_suggestion))
                } else {
                    // Otherwise—sorry, this is kind of gross—we need to infer the
                    // place to splice in the `'_, ` from the generics that do exist.
                    let first_generic_span = first_generic_span
                        .expect("already checked that type args or bindings exist");
                    (false, first_generic_span.shrink_to_lo(), format!("{}, ", anon_lt_suggestion))
                };
                self.sess.buffer_lint_with_diagnostic(
                    ELIDED_LIFETIMES_IN_PATHS,
                    CRATE_NODE_ID,
                    path_span,
                    "hidden lifetime parameters in types are deprecated",
                    builtin::BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                        expected_lifetimes, path_span, incl_angl_brckt, insertion_span, suggestion
                    )
                );
            }
        }

        let def = self.expect_full_def(segment.id);
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
            Some(id.node_id),
            Some(id.hir_id),
            Some(def),
            generic_args,
            infer_types,
        )
    }

    fn lower_angle_bracketed_parameter_data(
        &mut self,
        data: &AngleBracketedArgs,
        param_mode: ParamMode,
        mut itctx: ImplTraitContext<'_>,
    ) -> (hir::GenericArgs, bool) {
        let &AngleBracketedArgs { ref args, ref bindings, .. } = data;
        let has_types = args.iter().any(|arg| match arg {
            ast::GenericArg::Type(_) => true,
            _ => false,
        });
        (hir::GenericArgs {
            args: args.iter().map(|a| self.lower_generic_arg(a, itctx.reborrow())).collect(),
            bindings: bindings.iter().map(|b| self.lower_ty_binding(b, itctx.reborrow())).collect(),
            parenthesized: false,
        },
        !has_types && param_mode == ParamMode::Optional)
    }

    fn lower_parenthesized_parameter_data(
        &mut self,
        data: &ParenthesizedArgs,
    ) -> (hir::GenericArgs, bool) {
        // Switch to `PassThrough` mode for anonymous lifetimes: this
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
                    let LoweredNodeId { node_id: _, hir_id } = this.next_id();
                    hir::Ty { node: hir::TyKind::Tup(tys), hir_id, span }
                };
                let LoweredNodeId { node_id, hir_id } = this.next_id();

                (
                    hir::GenericArgs {
                        args: hir_vec![GenericArg::Type(mk_tup(this, inputs, span))],
                        bindings: hir_vec![
                            hir::TypeBinding {
                                id: node_id,
                                hir_id,
                                ident: Ident::from_str(FN_OUTPUT_NAME),
                                ty: output
                                    .as_ref()
                                    .map(|ty| this.lower_ty(&ty, ImplTraitContext::disallowed()))
                                    .unwrap_or_else(|| P(mk_tup(this, hir::HirVec::new(), span))),
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

    fn lower_local(&mut self, l: &Local) -> (hir::Local, SmallVec<[hir::ItemId; 1]>) {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(l.id);
        let mut ids = SmallVec::<[hir::ItemId; 1]>::new();
        if self.sess.features_untracked().impl_trait_in_bindings {
            if let Some(ref ty) = l.ty {
                let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                visitor.visit_ty(ty);
            }
        }
        let parent_def_id = DefId::local(self.current_hir_id_owner.last().unwrap().0);
        (hir::Local {
            id: node_id,
            hir_id,
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
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(arg.id);
        hir::Arg {
            id: node_id,
            hir_id,
            pat: self.lower_pat(&arg.pat),
        }
    }

    fn lower_fn_args_to_names(&mut self, decl: &FnDecl) -> hir::HirVec<Ident> {
        decl.inputs
            .iter()
            .map(|arg| match arg.pat.node {
                PatKind::Ident(_, ident, _) => ident,
                _ => Ident::new(keywords::Invalid.name(), arg.pat.span),
            })
            .collect()
    }

    // Lowers a function declaration.
    //
    // decl: the unlowered (ast) function declaration.
    // fn_def_id: if `Some`, impl Trait arguments are lowered into generic parameters on the
    //      given DefId, otherwise impl Trait is disallowed. Must be `Some` if
    //      make_ret_async is also `Some`.
    // impl_trait_return_allow: determines whether impl Trait can be used in return position.
    //      This guards against trait declarations and implementations where impl Trait is
    //      disallowed.
    // make_ret_async: if `Some`, converts `-> T` into `-> impl Future<Output = T>` in the
    //      return type. This is used for `async fn` declarations. The `NodeId` is the id of the
    //      return type impl Trait item.
    fn lower_fn_decl(
        &mut self,
        decl: &FnDecl,
        mut in_band_ty_params: Option<(DefId, &mut Vec<hir::GenericParam>)>,
        impl_trait_return_allow: bool,
        make_ret_async: Option<NodeId>,
    ) -> P<hir::FnDecl> {
        let inputs = decl.inputs
            .iter()
            .map(|arg| {
                if let Some((_, ref mut ibty)) = in_band_ty_params {
                    self.lower_ty_direct(&arg.ty, ImplTraitContext::Universal(ibty))
                } else {
                    self.lower_ty_direct(&arg.ty, ImplTraitContext::disallowed())
                }
            })
            .collect::<HirVec<_>>();

        let output = if let Some(ret_id) = make_ret_async {
            self.lower_async_fn_ret_ty(
                &inputs,
                &decl.output,
                in_band_ty_params.expect("make_ret_async but no fn_def_id").0,
                ret_id,
            )
        } else {
            match decl.output {
                FunctionRetTy::Ty(ref ty) => match in_band_ty_params {
                    Some((def_id, _)) if impl_trait_return_allow => {
                        hir::Return(self.lower_ty(ty,
                            ImplTraitContext::Existential(Some(def_id))))
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
            variadic: decl.variadic,
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

    // Transform `-> T` into `-> impl Future<Output = T>` for `async fn`
    //
    // fn_span: the span of the async function declaration. Used for error reporting.
    // inputs: lowered types of arguments to the function. Used to collect lifetimes.
    // output: unlowered output type (`T` in `-> T`)
    // fn_def_id: DefId of the parent function. Used to create child impl trait definition.
    fn lower_async_fn_ret_ty(
        &mut self,
        inputs: &[hir::Ty],
        output: &FunctionRetTy,
        fn_def_id: DefId,
        return_impl_trait_id: NodeId,
    ) -> hir::FunctionRetTy {
        // Get lifetimes used in the input arguments to the function. Our output type must also
        // have the same lifetime.
        // FIXME(cramertj): multiple different lifetimes are not allowed because
        // `impl Trait + 'a + 'b` doesn't allow for capture `'a` and `'b` where neither is a subset
        // of the other. We really want some new lifetime that is a subset of all input lifetimes,
        // but that doesn't exist at the moment.

        struct AsyncFnLifetimeCollector<'r, 'a: 'r> {
            context: &'r mut LoweringContext<'a>,
            // Lifetimes bound by HRTB.
            currently_bound_lifetimes: Vec<hir::LifetimeName>,
            // Whether to count elided lifetimes.
            // Disabled inside of `Fn` or `fn` syntax.
            collect_elided_lifetimes: bool,
            // The lifetime found.
            // Multiple different or elided lifetimes cannot appear in async fn for now.
            output_lifetime: Option<(hir::LifetimeName, Span)>,
        }

        impl<'r, 'a: 'r, 'v> hir::intravisit::Visitor<'v> for AsyncFnLifetimeCollector<'r, 'a> {
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
                if let &hir::TyKind::BareFn(_) = &t.node {
                    let old_collect_elided_lifetimes = self.collect_elided_lifetimes;
                    self.collect_elided_lifetimes = false;

                    // Record the "stack height" of `for<'a>` lifetime bindings
                    // to be able to later fully undo their introduction.
                    let old_len = self.currently_bound_lifetimes.len();
                    hir::intravisit::walk_ty(self, t);
                    self.currently_bound_lifetimes.truncate(old_len);

                    self.collect_elided_lifetimes = old_collect_elided_lifetimes;
                } else {
                    hir::intravisit::walk_ty(self, t);
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
                 // Record the introduction of 'a in `for<'a> ...`
                if let hir::GenericParamKind::Lifetime { .. } = param.kind {
                    // Introduce lifetimes one at a time so that we can handle
                    // cases like `fn foo<'d>() -> impl for<'a, 'b: 'a, 'c: 'b + 'd>`
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
                            // `abstract type Foo<'_>: SomeTrait<'_>;`
                            hir::LifetimeName::Underscore
                        } else {
                            return;
                        }
                    }
                    hir::LifetimeName::Param(_) => lifetime.name,
                    hir::LifetimeName::Error | hir::LifetimeName::Static => return,
                };

                if !self.currently_bound_lifetimes.contains(&name) {
                    if let Some((current_lt_name, current_lt_span)) = self.output_lifetime {
                        // We don't currently have a reliable way to desugar `async fn` with
                        // multiple potentially unrelated input lifetimes into
                        // `-> impl Trait + 'lt`, so we report an error in this case.
                        if current_lt_name != name {
                            struct_span_err!(
                                self.context.sess,
                                MultiSpan::from_spans(vec![current_lt_span, lifetime.span]),
                                E0709,
                                "multiple different lifetimes used in arguments of `async fn`",
                            )
                                .span_label(current_lt_span, "first lifetime here")
                                .span_label(lifetime.span, "different lifetime here")
                                .help("`async fn` can only accept borrowed values \
                                      with identical lifetimes")
                                .emit()
                        } else if current_lt_name.is_elided() && name.is_elided() {
                            struct_span_err!(
                                self.context.sess,
                                MultiSpan::from_spans(vec![current_lt_span, lifetime.span]),
                                E0707,
                                "multiple elided lifetimes used in arguments of `async fn`",
                            )
                                .span_label(current_lt_span, "first lifetime here")
                                .span_label(lifetime.span, "different lifetime here")
                                .help("consider giving these arguments named lifetimes")
                                .emit()
                        }
                    } else {
                        self.output_lifetime = Some((name, lifetime.span));
                    }
                }
            }
        }

        let bound_lifetime = {
            let mut lifetime_collector = AsyncFnLifetimeCollector {
                context: self,
                currently_bound_lifetimes: Vec::new(),
                collect_elided_lifetimes: true,
                output_lifetime: None,
            };

            for arg in inputs {
                hir::intravisit::walk_ty(&mut lifetime_collector, arg);
            }
            lifetime_collector.output_lifetime
        };

        let span = match output {
            FunctionRetTy::Ty(ty) => ty.span,
            FunctionRetTy::Default(span) => *span,
        };

        let impl_trait_ty = self.lower_existential_impl_trait(
            span, Some(fn_def_id), return_impl_trait_id, |this| {
            let output_ty = match output {
                FunctionRetTy::Ty(ty) => {
                    this.lower_ty(ty, ImplTraitContext::Existential(Some(fn_def_id)))
                }
                FunctionRetTy::Default(span) => {
                    let LoweredNodeId { node_id: _, hir_id } = this.next_id();
                    P(hir::Ty {
                        hir_id,
                        node: hir::TyKind::Tup(hir_vec![]),
                        span: *span,
                    })
                }
            };

            // "<Output = T>"
            let LoweredNodeId { node_id, hir_id } = this.next_id();
            let future_params = P(hir::GenericArgs {
                args: hir_vec![],
                bindings: hir_vec![hir::TypeBinding {
                    ident: Ident::from_str(FN_OUTPUT_NAME),
                    ty: output_ty,
                    id: node_id,
                    hir_id,
                    span,
                }],
                parenthesized: false,
            });

            let future_path =
                this.std_path(span, &["future", "Future"], Some(future_params), false);

            let LoweredNodeId { node_id, hir_id } = this.next_id();
            let mut bounds = vec![
                hir::GenericBound::Trait(
                    hir::PolyTraitRef {
                        trait_ref: hir::TraitRef {
                            path: future_path,
                            ref_id: node_id,
                            hir_ref_id: hir_id,
                        },
                        bound_generic_params: hir_vec![],
                        span,
                    },
                    hir::TraitBoundModifier::None
                ),
            ];

            if let Some((name, span)) = bound_lifetime {
                let LoweredNodeId { node_id: _, hir_id } = this.next_id();
                bounds.push(hir::GenericBound::Outlives(
                    hir::Lifetime { hir_id, name, span }));
            }

            hir::HirVec::from(bounds)
        });

        let LoweredNodeId { node_id: _, hir_id } = self.next_id();
        let impl_trait_ty = P(hir::Ty {
            node: impl_trait_ty,
            span,
            hir_id,
        });

        hir::FunctionRetTy::Return(impl_trait_ty)
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
            ident if ident.name == keywords::StaticLifetime.name() =>
                self.new_named_lifetime(l.id, span, hir::LifetimeName::Static),
            ident if ident.name == keywords::UnderscoreLifetime.name() =>
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
        let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(id);

        hir::Lifetime {
            hir_id,
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
                let ident = if param.ident.name == keywords::SelfUpper.name() {
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
                        self.lower_ty(x, ImplTraitContext::disallowed())
                    }),
                    synthetic: param.attrs.iter()
                                          .filter(|attr| attr.check_name("rustc_synthetic"))
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

        let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(param.id);

        hir::GenericParam {
            hir_id,
            name,
            span: param.ident.span,
            pure_wrt_drop: attr::contains_name(&param.attrs, "may_dangle"),
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
        // FIXME: this could probably be done with less rightward drift. Also looks like two control
        //        paths where report_error is called are also the only paths that advance to after
        //        the match statement, so the error reporting could probably just be moved there.
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
                                if let Some(Def::TyParam(def_id)) = self.resolver
                                    .get_resolution(bound_pred.bounded_ty.id)
                                    .map(|d| d.base_def())
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
                let LoweredNodeId { node_id: _, hir_id } = this.lower_node_id(wc.id);

                hir::WhereClause {
                    hir_id,
                    predicates: wc.predicates
                        .iter()
                        .map(|predicate| this.lower_where_predicate(predicate))
                        .collect(),
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
                let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(id);

                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    hir_id,
                    lhs_ty: self.lower_ty(lhs_ty, ImplTraitContext::disallowed()),
                    rhs_ty: self.lower_ty(rhs_ty, ImplTraitContext::disallowed()),
                    span,
                })
            },
        }
    }

    fn lower_variant_data(&mut self, vdata: &VariantData) -> hir::VariantData {
        match *vdata {
            VariantData::Struct(ref fields, id) => {
                let LoweredNodeId { node_id, hir_id } = self.lower_node_id(id);

                hir::VariantData::Struct(
                    fields
                        .iter()
                        .enumerate()
                        .map(|f| self.lower_struct_field(f))
                        .collect(),
                    node_id,
                    hir_id,
                )
            },
            VariantData::Tuple(ref fields, id) => {
                let LoweredNodeId { node_id, hir_id } = self.lower_node_id(id);

                hir::VariantData::Tuple(
                    fields
                        .iter()
                        .enumerate()
                        .map(|f| self.lower_struct_field(f))
                        .collect(),
                    node_id,
                    hir_id,
                )
            },
            VariantData::Unit(id) => {
                let LoweredNodeId { node_id, hir_id } = self.lower_node_id(id);

                hir::VariantData::Unit(node_id, hir_id)
            },
        }
    }

    fn lower_trait_ref(&mut self, p: &TraitRef, itctx: ImplTraitContext<'_>) -> hir::TraitRef {
        let path = match self.lower_qpath(p.ref_id, &None, &p.path, ParamMode::Explicit, itctx) {
            hir::QPath::Resolved(None, path) => path.and_then(|path| path),
            qpath => bug!("lower_trait_ref: unexpected QPath `{:?}`", qpath),
        };
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(p.ref_id);
        hir::TraitRef {
            path,
            ref_id: node_id,
            hir_ref_id: hir_id,
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
        let trait_ref = self.with_parent_impl_lifetime_defs(
            &bound_generic_params,
            |this| this.lower_trait_ref(&p.trait_ref, itctx),
        );

        hir::PolyTraitRef {
            bound_generic_params,
            trait_ref,
            span: p.span,
        }
    }

    fn lower_struct_field(&mut self, (index, f): (usize, &StructField)) -> hir::StructField {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(f.id);

        hir::StructField {
            span: f.span,
            id: node_id,
            hir_id,
            ident: match f.ident {
                Some(ident) => ident,
                // FIXME(jseyfried): positional field hygiene
                None => Ident::new(Symbol::intern(&index.to_string()), f.span),
            },
            vis: self.lower_visibility(&f.vis, None),
            ty: self.lower_ty(&f.ty, ImplTraitContext::disallowed()),
            attrs: self.lower_attrs(&f.attrs),
        }
    }

    fn lower_field(&mut self, f: &Field) -> hir::Field {
        let LoweredNodeId { node_id, hir_id } = self.next_id();

        hir::Field {
            id: node_id,
            hir_id,
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

    fn lower_block(&mut self, b: &Block, targeted_by_break: bool) -> P<hir::Block> {
        let mut expr = None;

        let mut stmts = vec![];

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

        let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(b.id);

        P(hir::Block {
            hir_id,
            stmts: stmts.into(),
            expr,
            rules: self.lower_block_check_mode(&b.rules),
            span: b.span,
            targeted_by_break,
        })
    }

    fn lower_async_body(
        &mut self,
        decl: &FnDecl,
        asyncness: IsAsync,
        body: &Block,
    ) -> hir::BodyId {
        self.lower_body(Some(decl), |this| {
            if let IsAsync::Async { closure_id, .. } = asyncness {
                let async_expr = this.make_async_expr(
                    CaptureBy::Value, closure_id, None,
                    |this| {
                        let body = this.lower_block(body, false);
                        this.expr_block(body, ThinVec::new())
                    });
                this.expr(body.span, async_expr, ThinVec::new())
            } else {
                let body = this.lower_block(body, false);
                this.expr_block(body, ThinVec::new())
            }
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
                // Start with an empty prefix
                let prefix = Path {
                    segments: vec![],
                    span: use_tree.span,
                };

                self.lower_use_tree(use_tree, &prefix, id, vis, ident, attrs)
            }
            ItemKind::Static(ref t, m, ref e) => {
                let value = self.lower_body(None, |this| this.lower_expr(e));
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
                    value,
                )
            }
            ItemKind::Const(ref t, ref e) => {
                let value = self.lower_body(None, |this| this.lower_expr(e));
                hir::ItemKind::Const(
                    self.lower_ty(
                        t,
                        if self.sess.features_untracked().impl_trait_in_bindings {
                            ImplTraitContext::Existential(None)
                        } else {
                            ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                        }
                    ),
                    value
                )
            }
            ItemKind::Fn(ref decl, header, ref generics, ref body) => {
                let fn_def_id = self.resolver.definitions().local_def_id(id);
                self.with_new_scopes(|this| {
                    // Note: we don't need to change the return type from `T` to
                    // `impl Future<Output = T>` here because lower_body
                    // only cares about the input argument patterns in the function
                    // declaration (decl), not the return types.
                    let body_id = this.lower_async_body(decl, header.asyncness.node, body);

                    let (generics, fn_decl) = this.add_in_band_defs(
                        generics,
                        fn_def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, idty| this.lower_fn_decl(
                            decl,
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
            ItemKind::Existential(ref b, ref generics) => hir::ItemKind::Existential(hir::ExistTy {
                generics: self.lower_generics(generics, ImplTraitContext::disallowed()),
                bounds: self.lower_param_bounds(b, ImplTraitContext::disallowed()),
                impl_trait_fn: None,
            }),
            ItemKind::Enum(ref enum_definition, ref generics) => hir::ItemKind::Enum(
                hir::EnumDef {
                    variants: enum_definition
                        .variants
                        .iter()
                        .map(|x| self.lower_variant(x))
                        .collect(),
                },
                self.lower_generics(generics, ImplTraitContext::disallowed()),
            ),
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
                let (generics, (trait_ref, lowered_ty)) = self.add_in_band_defs(
                    ast_generics,
                    def_id,
                    AnonymousLifetimeMode::CreateParameter,
                    |this, _| {
                        let trait_ref = trait_ref.as_ref().map(|trait_ref| {
                            this.lower_trait_ref(trait_ref, ImplTraitContext::disallowed())
                        });

                        if let Some(ref trait_ref) = trait_ref {
                            if let Def::Trait(def_id) = trait_ref.path.def {
                                this.trait_impls.entry(def_id).or_default().push(id);
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
            ItemKind::MacroDef(..) | ItemKind::Mac(..) => panic!("Shouldn't still be around"),
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
                    && path.segments.last().unwrap().ident.name == keywords::SelfLower.name()
                {
                    let _ = path.segments.pop();
                    if rename.is_none() {
                        *ident = path.segments.last().unwrap().ident;
                    }
                }

                let parent_def_index = self.current_hir_id_owner.last().unwrap().0;
                let mut defs = self.expect_full_def_from_use(id);
                // We want to return *something* from this function, so hold onto the first item
                // for later.
                let ret_def = defs.next().unwrap_or(Def::Err);

                // Here, we are looping over namespaces, if they exist for the definition
                // being imported. We only handle type and value namespaces because we
                // won't be dealing with macros in the rest of the compiler.
                // Essentially a single `use` which imports two names is desugared into
                // two imports.
                for (def, &new_node_id) in defs.zip([id1, id2].iter()) {
                    let vis = vis.clone();
                    let ident = ident.clone();
                    let mut path = path.clone();
                    for seg in &mut path.segments {
                        seg.id = self.sess.next_node_id();
                    }
                    let span = path.span;
                    self.resolver.definitions().create_def_with_parent(
                        parent_def_index,
                        new_node_id,
                        DefPathData::Misc,
                        DefIndexAddressSpace::High,
                        Mark::root(),
                        span);
                    self.allocate_hir_id_counter(new_node_id, &path);

                    self.with_hir_id_owner(new_node_id, |this| {
                        let new_id = this.lower_node_id(new_node_id);
                        let path =
                            this.lower_path_extra(def, &path, ParamMode::Explicit, None);
                        let item = hir::ItemKind::Use(P(path), hir::UseKind::Single);
                        let vis_kind = match vis.node {
                            hir::VisibilityKind::Public => hir::VisibilityKind::Public,
                            hir::VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
                            hir::VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
                            hir::VisibilityKind::Restricted { ref path, id: _, hir_id: _ } => {
                                let id = this.next_id();
                                let path = this.renumber_segment_ids(path);
                                hir::VisibilityKind::Restricted {
                                    path,
                                    id: id.node_id,
                                    hir_id: id.hir_id,
                                }
                            }
                        };
                        let vis = respan(vis.span, vis_kind);

                        this.insert_item(
                            new_id.node_id,
                            hir::Item {
                                id: new_id.node_id,
                                hir_id: new_id.hir_id,
                                ident,
                                attrs: attrs.clone(),
                                node: item,
                                vis,
                                span,
                            },
                        );
                    });
                }

                let path =
                    P(self.lower_path_extra(ret_def, &path, ParamMode::Explicit, None));
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
                    self.allocate_hir_id_counter(id, &use_tree);

                    let LoweredNodeId {
                        node_id: new_id,
                        hir_id: new_hir_id,
                    } = self.lower_node_id(id);

                    let mut vis = vis.clone();
                    let mut ident = ident.clone();
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
                    self.with_hir_id_owner(new_id, |this| {
                        let item = this.lower_use_tree(use_tree,
                                                       &prefix,
                                                       new_id,
                                                       &mut vis,
                                                       &mut ident,
                                                       attrs);

                        let vis_kind = match vis.node {
                            hir::VisibilityKind::Public => hir::VisibilityKind::Public,
                            hir::VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
                            hir::VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
                            hir::VisibilityKind::Restricted { ref path, id: _, hir_id: _ } => {
                                let id = this.next_id();
                                let path = this.renumber_segment_ids(path);
                                hir::VisibilityKind::Restricted {
                                    path: path,
                                    id: id.node_id,
                                    hir_id: id.hir_id,
                                }
                            }
                        };
                        let vis = respan(vis.span, vis_kind);

                        this.insert_item(
                            new_id,
                            hir::Item {
                                id: new_id,
                                hir_id: new_hir_id,
                                ident,
                                attrs: attrs.clone(),
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

                let def = self.expect_full_def_from_use(id).next().unwrap_or(Def::Err);
                let path = P(self.lower_path_extra(def, &prefix, ParamMode::Explicit, None));
                hir::ItemKind::Use(path, hir::UseKind::ListStem)
            }
        }
    }

    /// Paths like the visibility path in `pub(super) use foo::{bar, baz}` are repeated
    /// many times in the HIR tree; for each occurrence, we need to assign distinct
    /// `NodeId`s. (See, e.g., #56128.)
    fn renumber_segment_ids(&mut self, path: &P<hir::Path>) -> P<hir::Path> {
        debug!("renumber_segment_ids(path = {:?})", path);
        let mut path = path.clone();
        for seg in path.segments.iter_mut() {
            if seg.id.is_some() {
                let next_id = self.next_id();
                seg.id = Some(next_id.node_id);
                seg.hir_id = Some(next_id.hir_id);
            }
        }
        path
    }

    fn lower_trait_item(&mut self, i: &TraitItem) -> hir::TraitItem {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(i.id);
        let trait_item_def_id = self.resolver.definitions().local_def_id(node_id);

        let (generics, node) = match i.node {
            TraitItemKind::Const(ref ty, ref default) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::TraitItemKind::Const(
                    self.lower_ty(ty, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_body(None, |this| this.lower_expr(x))),
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
                let body_id = self.lower_body(Some(&sig.decl), |this| {
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
            TraitItemKind::Type(ref bounds, ref default) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::TraitItemKind::Type(
                    self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_ty(x, ImplTraitContext::disallowed())),
                ),
            ),
            TraitItemKind::Macro(..) => panic!("Shouldn't exist any more"),
        };

        hir::TraitItem {
            id: node_id,
            hir_id,
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
                (hir::AssociatedItemKind::Const, default.is_some())
            }
            TraitItemKind::Type(_, ref default) => {
                (hir::AssociatedItemKind::Type, default.is_some())
            }
            TraitItemKind::Method(ref sig, ref default) => (
                hir::AssociatedItemKind::Method {
                    has_self: sig.decl.has_self(),
                },
                default.is_some(),
            ),
            TraitItemKind::Macro(..) => unimplemented!(),
        };
        hir::TraitItemRef {
            id: hir::TraitItemId { node_id: i.id },
            ident: i.ident,
            span: i.span,
            defaultness: self.lower_defaultness(Defaultness::Default, has_default),
            kind,
        }
    }

    fn lower_impl_item(&mut self, i: &ImplItem) -> hir::ImplItem {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(i.id);
        let impl_item_def_id = self.resolver.definitions().local_def_id(node_id);

        let (generics, node) = match i.node {
            ImplItemKind::Const(ref ty, ref expr) => {
                let body_id = self.lower_body(None, |this| this.lower_expr(expr));
                (
                    self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                    hir::ImplItemKind::Const(
                        self.lower_ty(ty, ImplTraitContext::disallowed()),
                        body_id,
                    ),
                )
            }
            ImplItemKind::Method(ref sig, ref body) => {
                let body_id = self.lower_async_body(&sig.decl, sig.header.asyncness.node, body);
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
            ImplItemKind::Macro(..) => panic!("Shouldn't exist any more"),
        };

        hir::ImplItem {
            id: node_id,
            hir_id,
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
            id: hir::ImplItemId { node_id: i.id },
            ident: i.ident,
            span: i.span,
            vis: self.lower_visibility(&i.vis, Some(i.id)),
            defaultness: self.lower_defaultness(i.defaultness, true /* [1] */),
            kind: match i.node {
                ImplItemKind::Const(..) => hir::AssociatedItemKind::Const,
                ImplItemKind::Type(..) => hir::AssociatedItemKind::Type,
                ImplItemKind::Existential(..) => hir::AssociatedItemKind::Existential,
                ImplItemKind::Method(ref sig, _) => hir::AssociatedItemKind::Method {
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
        match i.node {
            ItemKind::Use(ref use_tree) => {
                let mut vec = smallvec![hir::ItemId { id: i.id }];
                self.lower_item_id_use_tree(use_tree, i.id, &mut vec);
                vec
            }
            ItemKind::MacroDef(..) => SmallVec::new(),
            ItemKind::Fn(..) |
            ItemKind::Impl(.., None, _, _) => smallvec![hir::ItemId { id: i.id }],
            ItemKind::Static(ref ty, ..) => {
                let mut ids = smallvec![hir::ItemId { id: i.id }];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            ItemKind::Const(ref ty, ..) => {
                let mut ids = smallvec![hir::ItemId { id: i.id }];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            _ => smallvec![hir::ItemId { id: i.id }],
        }
    }

    fn lower_item_id_use_tree(&mut self,
                              tree: &UseTree,
                              base_id: NodeId,
                              vec: &mut SmallVec<[hir::ItemId; 1]>)
    {
        match tree.kind {
            UseTreeKind::Nested(ref nested_vec) => for &(ref nested, id) in nested_vec {
                vec.push(hir::ItemId { id });
                self.lower_item_id_use_tree(nested, id, vec);
            },
            UseTreeKind::Glob => {}
            UseTreeKind::Simple(_, id1, id2) => {
                for (_, &id) in self.expect_full_def_from_use(base_id)
                                    .skip(1)
                                    .zip([id1, id2].iter())
                {
                    vec.push(hir::ItemId { id });
                }
            },
        }
    }

    pub fn lower_item(&mut self, i: &Item) -> Option<hir::Item> {
        let mut ident = i.ident;
        let mut vis = self.lower_visibility(&i.vis, None);
        let attrs = self.lower_attrs(&i.attrs);
        if let ItemKind::MacroDef(ref def) = i.node {
            if !def.legacy || attr::contains_name(&i.attrs, "macro_export") ||
                              attr::contains_name(&i.attrs, "rustc_doc_only_macro") {
                let body = self.lower_token_stream(def.stream());
                let hir_id = self.lower_node_id(i.id).hir_id;
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

        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(i.id);

        Some(hir::Item {
            id: node_id,
            hir_id,
            ident,
            attrs,
            node,
            vis,
            span: i.span,
        })
    }

    fn lower_foreign_item(&mut self, i: &ForeignItem) -> hir::ForeignItem {
        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(i.id);
        let def_id = self.resolver.definitions().local_def_id(node_id);
        hir::ForeignItem {
            id: node_id,
            hir_id,
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
                        self.lower_ty(t, ImplTraitContext::disallowed()), m)
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
                match self.resolver.get_resolution(p.id).map(|d| d.base_def()) {
                    // `None` can occur in body-less function signatures
                    def @ None | def @ Some(Def::Local(_)) => {
                        let canonical_id = match def {
                            Some(Def::Local(id)) => id,
                            _ => p.id,
                        };
                        let hir_id = self.lower_node_id(canonical_id).hir_id;
                        hir::PatKind::Binding(
                            self.lower_binding_mode(binding_mode),
                            canonical_id,
                            hir_id,
                            ident,
                            sub.as_ref().map(|x| self.lower_pat(x)),
                        )
                    }
                    Some(def) => hir::PatKind::Path(hir::QPath::Resolved(
                        None,
                        P(hir::Path {
                            span: ident.span,
                            def,
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
                        let LoweredNodeId { node_id, hir_id } = self.next_id();

                        Spanned {
                            span: f.span,
                            node: hir::FieldPat {
                                id: node_id,
                                hir_id,
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

        let LoweredNodeId { node_id, hir_id } = self.lower_node_id(p.id);
        P(hir::Pat {
            id: node_id,
            hir_id,
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
            let LoweredNodeId { node_id, hir_id } = this.lower_node_id(c.id);
            hir::AnonConst {
                id: node_id,
                hir_id,
                body: this.lower_body(None, |this| this.lower_expr(&c.value)),
            }
        })
    }

    fn lower_expr(&mut self, e: &Expr) -> hir::Expr {
        let kind = match e.node {
            ExprKind::Box(ref inner) => hir::ExprKind::Box(P(self.lower_expr(inner))),
            ExprKind::ObsoleteInPlace(..) => {
                self.sess.abort_if_errors();
                span_bug!(e.span, "encountered ObsoleteInPlace expr during lowering");
            }
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
            ExprKind::Lit(ref l) => hir::ExprKind::Lit((*l).clone()),
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
            // More complicated than you might expect because the else branch
            // might be `if let`.
            ExprKind::If(ref cond, ref blk, ref else_opt) => {
                let else_opt = else_opt.as_ref().map(|els| {
                    match els.node {
                        ExprKind::IfLet(..) => {
                            // Wrap the `if let` expr in a block.
                            let span = els.span;
                            let els = P(self.lower_expr(els));
                            let LoweredNodeId { node_id: _, hir_id } = self.next_id();
                            let blk = P(hir::Block {
                                stmts: hir_vec![],
                                expr: Some(els),
                                hir_id,
                                rules: hir::DefaultBlock,
                                span,
                                targeted_by_break: false,
                            });
                            P(self.expr_block(blk, ThinVec::new()))
                        }
                        _ => P(self.lower_expr(els)),
                    }
                });

                let then_blk = self.lower_block(blk, false);
                let then_expr = self.expr_block(then_blk, ThinVec::new());

                hir::ExprKind::If(P(self.lower_expr(cond)), P(then_expr), else_opt)
            }
            ExprKind::While(ref cond, ref body, opt_label) => self.with_loop_scope(e.id, |this| {
                hir::ExprKind::While(
                    this.with_loop_condition_scope(|this| P(this.lower_expr(cond))),
                    this.lower_block(body, false),
                    this.lower_label(opt_label),
                )
            }),
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
                        Some(vec![
                            Symbol::intern("try_trait"),
                        ].into()),
                    );
                    let mut block = this.lower_block(body, true).into_inner();
                    let tail = block.expr.take().map_or_else(
                        || {
                            let LoweredNodeId { node_id: _, hir_id } = this.next_id();
                            let span = this.sess.source_map().end_point(unstable_span);
                            hir::Expr {
                                span,
                                node: hir::ExprKind::Tup(hir_vec![]),
                                attrs: ThinVec::new(),
                                hir_id,
                            }
                        },
                        |x: P<hir::Expr>| x.into_inner(),
                    );
                    block.expr = Some(this.wrap_in_try_constructor(
                        "from_ok", tail, unstable_span));
                    hir::ExprKind::Block(P(block), None)
                })
            }
            ExprKind::Match(ref expr, ref arms) => hir::ExprKind::Match(
                P(self.lower_expr(expr)),
                arms.iter().map(|x| self.lower_arm(x)).collect(),
                hir::MatchSource::Normal,
            ),
            ExprKind::Async(capture_clause, closure_node_id, ref block) => {
                self.make_async_expr(capture_clause, closure_node_id, None, |this| {
                    this.with_new_scopes(|this| {
                        let block = this.lower_block(block, false);
                        this.expr_block(block, ThinVec::new())
                    })
                })
            }
            ExprKind::Closure(
                capture_clause, asyncness, movability, ref decl, ref body, fn_decl_span
            ) => {
                if let IsAsync::Async { closure_id, .. } = asyncness {
                    let outer_decl = FnDecl {
                        inputs: decl.inputs.clone(),
                        output: FunctionRetTy::Default(fn_decl_span),
                        variadic: false,
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
                        let body_id = this.lower_body(Some(&outer_decl), |this| {
                            let async_ret_ty = if let FunctionRetTy::Ty(ty) = &decl.output {
                                Some(&**ty)
                            } else { None };
                            let async_body = this.make_async_expr(
                                capture_clause, closure_id, async_ret_ty,
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
                        let mut is_generator = false;
                        let body_id = this.lower_body(Some(decl), |this| {
                            let e = this.lower_expr(body);
                            is_generator = this.is_generator;
                            e
                        });
                        let generator_option = if is_generator {
                            if !decl.inputs.is_empty() {
                                span_err!(
                                    this.sess,
                                    fn_decl_span,
                                    E0628,
                                    "generators cannot have explicit arguments"
                                );
                                this.sess.abort_if_errors();
                            }
                            Some(match movability {
                                Movability::Movable => hir::GeneratorMovability::Movable,
                                Movability::Static => hir::GeneratorMovability::Static,
                            })
                        } else {
                            if movability == Movability::Static {
                                span_err!(
                                    this.sess,
                                    fn_decl_span,
                                    E0697,
                                    "closures cannot be static"
                                );
                            }
                            None
                        };
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
                let ty_path = P(self.std_path(e.span, &["ops", "RangeInclusive"], None, false));
                let ty = P(self.ty_path(id, e.span, hir::QPath::Resolved(None, ty_path)));
                let new_seg = P(hir::PathSegment::from_ident(Ident::from_str("new")));
                let new_path = hir::QPath::TypeRelative(ty, new_seg);
                let new = P(self.expr(e.span, hir::ExprKind::Path(new_path), ThinVec::new()));
                hir::ExprKind::Call(new, hir_vec![e1, e2])
            }
            ExprKind::Range(ref e1, ref e2, lims) => {
                use syntax::ast::RangeLimits::*;

                let path = match (e1, e2, lims) {
                    (&None, &None, HalfOpen) => "RangeFull",
                    (&Some(..), &None, HalfOpen) => "RangeFrom",
                    (&None, &Some(..), HalfOpen) => "RangeTo",
                    (&Some(..), &Some(..), HalfOpen) => "Range",
                    (&None, &Some(..), Closed) => "RangeToInclusive",
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
                let struct_path = ["ops", path];
                let struct_path = self.std_path(e.span, &struct_path, None, is_unit);
                let struct_path = hir::QPath::Resolved(None, P(struct_path));

                let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(e.id);

                return hir::Expr {
                    hir_id,
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
                self.is_generator = true;
                let expr = opt_expr
                    .as_ref()
                    .map(|x| self.lower_expr(x))
                    .unwrap_or_else(||
                    self.expr(e.span, hir::ExprKind::Tup(hir_vec![]), ThinVec::new())
                );
                hir::ExprKind::Yield(P(expr))
            }

            ExprKind::Err => hir::ExprKind::Err,

            // Desugar `ExprIfLet`
            // from: `if let <pat> = <sub_expr> <body> [<else_opt>]`
            ExprKind::IfLet(ref pats, ref sub_expr, ref body, ref else_opt) => {
                // to:
                //
                //   match <sub_expr> {
                //     <pat> => <body>,
                //     _ => [<else_opt> | ()]
                //   }

                let mut arms = vec![];

                // `<pat> => <body>`
                {
                    let body = self.lower_block(body, false);
                    let body_expr = P(self.expr_block(body, ThinVec::new()));
                    let pats = pats.iter().map(|pat| self.lower_pat(pat)).collect();
                    arms.push(self.arm(pats, body_expr));
                }

                // _ => [<else_opt>|()]
                {
                    let wildcard_arm: Option<&Expr> = else_opt.as_ref().map(|p| &**p);
                    let wildcard_pattern = self.pat_wild(e.span);
                    let body = if let Some(else_expr) = wildcard_arm {
                        P(self.lower_expr(else_expr))
                    } else {
                        self.expr_tuple(e.span, hir_vec![])
                    };
                    arms.push(self.arm(hir_vec![wildcard_pattern], body));
                }

                let contains_else_clause = else_opt.is_some();

                let sub_expr = P(self.lower_expr(sub_expr));

                hir::ExprKind::Match(
                    sub_expr,
                    arms.into(),
                    hir::MatchSource::IfLetDesugar {
                        contains_else_clause,
                    },
                )
            }

            // Desugar `ExprWhileLet`
            // from: `[opt_ident]: while let <pat> = <sub_expr> <body>`
            ExprKind::WhileLet(ref pats, ref sub_expr, ref body, opt_label) => {
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
            }

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
                let head = self.lower_expr(head);
                let head_sp = head.span;
                let desugared_span = self.mark_span_with_reason(
                    CompilerDesugaringKind::ForLoop,
                    head_sp,
                    None,
                );

                let iter = self.str_to_ident("iter");

                let next_ident = self.str_to_ident("__next");
                let next_pat = self.pat_ident_binding_mode(
                    desugared_span,
                    next_ident,
                    hir::BindingAnnotation::Mutable,
                );

                // `::std::option::Option::Some(val) => next = val`
                let pat_arm = {
                    let val_ident = self.str_to_ident("val");
                    let val_pat = self.pat_ident(pat.span, val_ident);
                    let val_expr = P(self.expr_ident(pat.span, val_ident, val_pat.id));
                    let next_expr = P(self.expr_ident(pat.span, next_ident, next_pat.id));
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
                let iter_pat = self.pat_ident_binding_mode(
                    desugared_span,
                    iter,
                    hir::BindingAnnotation::Mutable
                );

                // `match ::std::iter::Iterator::next(&mut iter) { ... }`
                let match_expr = {
                    let iter = P(self.expr_ident(head_sp, iter, iter_pat.id));
                    let ref_mut_iter = self.expr_mut_addr_of(head_sp, iter);
                    let next_path = &["iter", "Iterator", "next"];
                    let next_path = P(self.expr_std_path(head_sp, next_path, None, ThinVec::new()));
                    let next_expr = P(self.expr_call(head_sp, next_path, hir_vec![ref_mut_iter]));
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
                let LoweredNodeId { node_id, hir_id } = self.next_id();
                let match_stmt = hir::Stmt {
                    id: node_id,
                    hir_id,
                    node: hir::StmtKind::Expr(match_expr),
                    span: head_sp,
                };

                let next_expr = P(self.expr_ident(head_sp, next_ident, next_pat.id));

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
                let LoweredNodeId { node_id, hir_id } = self.next_id();
                let body_stmt = hir::Stmt {
                    id: node_id,
                    hir_id,
                    node: hir::StmtKind::Expr(body_expr),
                    span: body.span,
                };

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
                let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(e.id);
                let loop_expr = P(hir::Expr {
                    hir_id,
                    node: loop_expr,
                    span: e.span,
                    attrs: ThinVec::new(),
                });

                // `mut iter => { ... }`
                let iter_arm = self.arm(hir_vec![iter_pat], loop_expr);

                // `match ::std::iter::IntoIterator::into_iter(<head>) { ... }`
                let into_iter_expr = {
                    let into_iter_path = &["iter", "IntoIterator", "into_iter"];
                    let into_iter = P(self.expr_std_path(
                            head_sp, into_iter_path, None, ThinVec::new()));
                    P(self.expr_call(head_sp, into_iter, hir_vec![head]))
                };

                let match_expr = P(self.expr_match(
                    head_sp,
                    into_iter_expr,
                    hir_vec![iter_arm],
                    hir::MatchSource::ForLoopDesugar,
                ));

                // `{ let _result = ...; _result }`
                // Underscore prevents an `unused_variables` lint if the head diverges.
                let result_ident = self.str_to_ident("_result");
                let (let_stmt, let_stmt_binding) =
                    self.stmt_let(e.span, false, result_ident, match_expr);

                let result = P(self.expr_ident(e.span, result_ident, let_stmt_binding));
                let block = P(self.block_all(e.span, hir_vec![let_stmt], Some(result)));
                // Add the attributes to the outer returned expr node.
                return self.expr_block(block, e.attrs.clone());
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
                    Some(vec![
                        Symbol::intern("try_trait")
                    ].into()),
                );

                // `Try::into_result(<expr>)`
                let discr = {
                    // expand <expr>
                    let sub_expr = self.lower_expr(sub_expr);

                    let path = &["ops", "Try", "into_result"];
                    let path = P(self.expr_std_path(
                            unstable_span, path, None, ThinVec::new()));
                    P(self.expr_call(e.span, path, hir_vec![sub_expr]))
                };

                // `#[allow(unreachable_code)]`
                let attr = {
                    // `allow(unreachable_code)`
                    let allow = {
                        let allow_ident = Ident::from_str("allow").with_span_pos(e.span);
                        let uc_ident = Ident::from_str("unreachable_code").with_span_pos(e.span);
                        let uc_nested = attr::mk_nested_word_item(uc_ident);
                        attr::mk_list_item(e.span, allow_ident, vec![uc_nested])
                    };
                    attr::mk_spanned_attr_outer(e.span, attr::mk_attr_id(), allow)
                };
                let attrs = vec![attr];

                // `Ok(val) => #[allow(unreachable_code)] val,`
                let ok_arm = {
                    let val_ident = self.str_to_ident("val");
                    let val_pat = self.pat_ident(e.span, val_ident);
                    let val_expr = P(self.expr_ident_with_attrs(
                        e.span,
                        val_ident,
                        val_pat.id,
                        ThinVec::from(attrs.clone()),
                    ));
                    let ok_pat = self.pat_ok(e.span, val_pat);

                    self.arm(hir_vec![ok_pat], val_expr)
                };

                // `Err(err) => #[allow(unreachable_code)]
                //              return Try::from_error(From::from(err)),`
                let err_arm = {
                    let err_ident = self.str_to_ident("err");
                    let err_local = self.pat_ident(e.span, err_ident);
                    let from_expr = {
                        let path = &["convert", "From", "from"];
                        let from = P(self.expr_std_path(
                                e.span, path, None, ThinVec::new()));
                        let err_expr = self.expr_ident(e.span, err_ident, err_local.id);

                        self.expr_call(e.span, from, hir_vec![err_expr])
                    };
                    let from_err_expr =
                        self.wrap_in_try_constructor("from_error", from_expr, unstable_span);
                    let thin_attrs = ThinVec::from(attrs);
                    let catch_scope = self.catch_scopes.last().map(|x| *x);
                    let ret_expr = if let Some(catch_node) = catch_scope {
                        P(self.expr(
                            e.span,
                            hir::ExprKind::Break(
                                hir::Destination {
                                    label: None,
                                    target_id: Ok(catch_node),
                                },
                                Some(from_err_expr),
                            ),
                            thin_attrs,
                        ))
                    } else {
                        P(self.expr(e.span, hir::ExprKind::Ret(Some(from_err_expr)), thin_attrs))
                    };

                    let err_pat = self.pat_err(e.span, err_local);
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

        let LoweredNodeId { node_id: _, hir_id } = self.lower_node_id(e.id);

        hir::Expr {
            hir_id,
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
                        let LoweredNodeId { node_id, hir_id } = self.next_id();

                        hir::Stmt {
                            id: node_id,
                            hir_id,
                            node: hir::StmtKind::Item(item_id),
                            span: s.span,
                        }
                    })
                    .collect();
                ids.push({
                    let LoweredNodeId { node_id, hir_id } = self.lower_node_id(s.id);

                    hir::Stmt {
                        id: node_id,
                        hir_id,
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
                        let LoweredNodeId { node_id, hir_id } = id.take()
                          .map(|id| self.lower_node_id(id))
                          .unwrap_or_else(|| self.next_id());

                        hir::Stmt {
                            id: node_id,
                            hir_id,
                            node: hir::StmtKind::Item(item_id),
                            span: s.span,
                        }
                    })
                    .collect();
            }
            StmtKind::Expr(ref e) => {
                let LoweredNodeId { node_id, hir_id } = self.lower_node_id(s.id);

                hir::Stmt {
                    id: node_id,
                    hir_id,
                    node: hir::StmtKind::Expr(P(self.lower_expr(e))),
                    span: s.span,
                }
            },
            StmtKind::Semi(ref e) => {
                let LoweredNodeId { node_id, hir_id } = self.lower_node_id(s.id);

                hir::Stmt {
                    id: node_id,
                    hir_id,
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
                let def = self.expect_full_def(id);
                hir::VisibilityKind::Restricted {
                    path: P(self.lower_path_extra(
                        def,
                        path,
                        ParamMode::Explicit,
                        explicit_owner,
                    )),
                    id: lowered_id.node_id,
                    hir_id: lowered_id.hir_id,
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
            attrs: hir_vec![],
            pats,
            guard: None,
            body: expr,
        }
    }

    fn field(&mut self, ident: Ident, expr: P<hir::Expr>, span: Span) -> hir::Field {
        let LoweredNodeId { node_id, hir_id } = self.next_id();

        hir::Field {
            id: node_id,
            hir_id,
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

    fn expr_ident(&mut self, span: Span, ident: Ident, binding: NodeId) -> hir::Expr {
        self.expr_ident_with_attrs(span, ident, binding, ThinVec::new())
    }

    fn expr_ident_with_attrs(
        &mut self,
        span: Span,
        ident: Ident,
        binding: NodeId,
        attrs: ThinVec<Attribute>,
    ) -> hir::Expr {
        let expr_path = hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            P(hir::Path {
                span,
                def: Def::Local(binding),
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
        components: &[&str],
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

    fn expr_tuple(&mut self, sp: Span, exprs: hir::HirVec<hir::Expr>) -> P<hir::Expr> {
        P(self.expr(sp, hir::ExprKind::Tup(exprs), ThinVec::new()))
    }

    fn expr(&mut self, span: Span, node: hir::ExprKind, attrs: ThinVec<Attribute>) -> hir::Expr {
        let LoweredNodeId { node_id: _, hir_id } = self.next_id();
        hir::Expr {
            hir_id,
            node,
            span,
            attrs,
        }
    }

    fn stmt_let_pat(
        &mut self,
        sp: Span,
        ex: Option<P<hir::Expr>>,
        pat: P<hir::Pat>,
        source: hir::LocalSource,
    ) -> hir::Stmt {
        let LoweredNodeId { node_id, hir_id } = self.next_id();

        let local = hir::Local {
            pat,
            ty: None,
            init: ex,
            id: node_id,
            hir_id,
            span: sp,
            attrs: ThinVec::new(),
            source,
        };

        let LoweredNodeId { node_id, hir_id } = self.next_id();
        hir::Stmt {
            id: node_id,
            hir_id,
            node: hir::StmtKind::Local(P(local)),
            span: sp
        }
    }

    fn stmt_let(
        &mut self,
        sp: Span,
        mutbl: bool,
        ident: Ident,
        ex: P<hir::Expr>,
    ) -> (hir::Stmt, NodeId) {
        let pat = if mutbl {
            self.pat_ident_binding_mode(sp, ident, hir::BindingAnnotation::Mutable)
        } else {
            self.pat_ident(sp, ident)
        };
        let pat_id = pat.id;
        (
            self.stmt_let_pat(sp, Some(ex), pat, hir::LocalSource::Normal),
            pat_id,
        )
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
        let LoweredNodeId { node_id: _, hir_id } = self.next_id();

        hir::Block {
            stmts,
            expr,
            hir_id,
            rules: hir::DefaultBlock,
            span,
            targeted_by_break: false,
        }
    }

    fn pat_ok(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &["result", "Result", "Ok"], hir_vec![pat])
    }

    fn pat_err(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &["result", "Result", "Err"], hir_vec![pat])
    }

    fn pat_some(&mut self, span: Span, pat: P<hir::Pat>) -> P<hir::Pat> {
        self.pat_std_enum(span, &["option", "Option", "Some"], hir_vec![pat])
    }

    fn pat_none(&mut self, span: Span) -> P<hir::Pat> {
        self.pat_std_enum(span, &["option", "Option", "None"], hir_vec![])
    }

    fn pat_std_enum(
        &mut self,
        span: Span,
        components: &[&str],
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

    fn pat_ident(&mut self, span: Span, ident: Ident) -> P<hir::Pat> {
        self.pat_ident_binding_mode(span, ident, hir::BindingAnnotation::Unannotated)
    }

    fn pat_ident_binding_mode(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> P<hir::Pat> {
        let LoweredNodeId { node_id, hir_id } = self.next_id();

        P(hir::Pat {
            id: node_id,
            hir_id,
            node: hir::PatKind::Binding(bm, node_id, hir_id, ident.with_span_pos(span), None),
            span,
        })
    }

    fn pat_wild(&mut self, span: Span) -> P<hir::Pat> {
        self.pat(span, hir::PatKind::Wild)
    }

    fn pat(&mut self, span: Span, pat: hir::PatKind) -> P<hir::Pat> {
        let LoweredNodeId { node_id, hir_id } = self.next_id();
        P(hir::Pat {
            id: node_id,
            hir_id,
            node: pat,
            span,
        })
    }

    /// Given suffix ["b","c","d"], returns path `::std::b::c::d` when
    /// `fld.cx.use_std`, and `::core::b::c::d` otherwise.
    /// The path is also resolved according to `is_value`.
    fn std_path(
        &mut self,
        span: Span,
        components: &[&str],
        params: Option<P<hir::GenericArgs>>,
        is_value: bool
    ) -> hir::Path {
        let mut path = self.resolver
            .resolve_str_path(span, self.crate_root, components, is_value);
        path.segments.last_mut().unwrap().args = params;


        for seg in path.segments.iter_mut() {
            if let Some(id) = seg.id {
                seg.id = Some(self.lower_node_id(id).node_id);
            }
        }
        path
    }

    fn ty_path(&mut self, id: LoweredNodeId, span: Span, qpath: hir::QPath) -> hir::Ty {
        let mut id = id;
        let node = match qpath {
            hir::QPath::Resolved(None, path) => {
                // Turn trait object paths into `TyKind::TraitObject` instead.
                match path.def {
                    Def::Trait(_) | Def::TraitAlias(_) => {
                        let principal = hir::PolyTraitRef {
                            bound_generic_params: hir::HirVec::new(),
                            trait_ref: hir::TraitRef {
                                path: path.and_then(|path| path),
                                ref_id: id.node_id,
                                hir_ref_id: id.hir_id,
                            },
                            span,
                        };

                        // The original ID is taken by the `PolyTraitRef`,
                        // so the `Ty` itself needs a different one.
                        id = self.next_id();
                        hir::TyKind::TraitObject(hir_vec![principal], self.elided_dyn_bound(span))
                    }
                    _ => hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                }
            }
            _ => hir::TyKind::Path(qpath),
        };
        hir::Ty {
            hir_id: id.hir_id,
            node,
            span,
        }
    }

    /// Invoked to create the lifetime argument for a type `&T`
    /// with no explicit lifetime.
    fn elided_ref_lifetime(&mut self, span: Span) -> hir::Lifetime {
        match self.anonymous_lifetime_mode {
            // Intercept when we are in an impl header and introduce an in-band lifetime.
            // Hence `impl Foo for &u32` becomes `impl<'f> Foo for &'f u32` for some fresh
            // `'f`.
            AnonymousLifetimeMode::CreateParameter => {
                let fresh_name = self.collect_fresh_in_band_lifetime(span);
                let LoweredNodeId { node_id: _, hir_id } = self.next_id();
                hir::Lifetime {
                    hir_id,
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
                self.next_id().node_id,
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
        match self.anonymous_lifetime_mode {
            // N.B., We intentionally ignore the create-parameter mode here
            // and instead "pass through" to resolve-lifetimes, which will then
            // report an error. This is because we don't want to support
            // impl elision for deprecated forms like
            //
            //     impl Foo for std::cell::Ref<u32> // note lack of '_
            AnonymousLifetimeMode::CreateParameter => {}

            AnonymousLifetimeMode::ReportError => {
                return (0..count)
                    .map(|_| self.new_error_lifetime(None, span))
                    .collect();
            }

            // This is the normal case.
            AnonymousLifetimeMode::PassThrough => {}
        }

        (0..count)
            .map(|_| self.new_implicit_lifetime(span))
            .collect()
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

        self.new_implicit_lifetime(span)
    }

    fn new_implicit_lifetime(&mut self, span: Span) -> hir::Lifetime {
        let LoweredNodeId { node_id: _, hir_id } = self.next_id();

        hir::Lifetime {
            hir_id,
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
        method: &'static str,
        e: hir::Expr,
        unstable_span: Span,
    ) -> P<hir::Expr> {
        let path = &["ops", "Try", method];
        let from_err = P(self.expr_std_path(unstable_span, path, None,
                                            ThinVec::new()));
        P(self.expr_call(e.span, from_err, hir_vec![e]))
    }
}

fn body_ids(bodies: &BTreeMap<hir::BodyId, hir::Body>) -> Vec<hir::BodyId> {
    // Sorting by span ensures that we get things in order within a
    // file, and also puts the files in a sensible order.
    let mut body_ids: Vec<_> = bodies.keys().cloned().collect();
    body_ids.sort_by_key(|b| bodies[b].value.span);
    body_ids
}
