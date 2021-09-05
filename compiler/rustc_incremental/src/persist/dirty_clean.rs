//! Debugging code to test fingerprints computed for query results.  For each node marked with
//! `#[rustc_clean]` we will compare the fingerprint from the current and from the previous
//! compilation session as appropriate:
//!
//! - `#[rustc_clean(cfg="rev2", except="typeck")]` if we are
//!   in `#[cfg(rev2)]`, then the fingerprints associated with
//!   `DepNode::typeck(X)` must be DIFFERENT (`X` is the `DefId` of the
//!   current node).
//! - `#[rustc_clean(cfg="rev2")]` same as above, except that the
//!   fingerprints must be the SAME (along with all other fingerprints).
//!
//! Errors are reported if we are in the suitable configuration but
//! the required condition is not met.

use rustc_ast::{self as ast, Attribute, NestedMetaItem};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::Node as HirNode;
use rustc_hir::{ImplItemKind, ItemKind as HirItem, TraitItemKind};
use rustc_middle::dep_graph::{label_strs, DepNode, DepNodeExt};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use std::iter::FromIterator;
use std::vec::Vec;

const EXCEPT: Symbol = sym::except;
const CFG: Symbol = sym::cfg;

// Base and Extra labels to build up the labels

/// For typedef, constants, and statics
const BASE_CONST: &[&str] = &[label_strs::type_of];

/// DepNodes for functions + methods
const BASE_FN: &[&str] = &[
    // Callers will depend on the signature of these items, so we better test
    label_strs::fn_sig,
    label_strs::generics_of,
    label_strs::predicates_of,
    label_strs::type_of,
    // And a big part of compilation (that we eventually want to cache) is type inference
    // information:
    label_strs::typeck,
];

/// DepNodes for Hir, which is pretty much everything
const BASE_HIR: &[&str] = &[
    // hir_owner and hir_owner_nodes should be computed for all nodes
    label_strs::hir_owner,
    label_strs::hir_owner_nodes,
];

/// `impl` implementation of struct/trait
const BASE_IMPL: &[&str] =
    &[label_strs::associated_item_def_ids, label_strs::generics_of, label_strs::impl_trait_ref];

/// DepNodes for mir_built/Optimized, which is relevant in "executable"
/// code, i.e., functions+methods
const BASE_MIR: &[&str] = &[label_strs::optimized_mir, label_strs::promoted_mir];

/// Struct, Enum and Union DepNodes
///
/// Note that changing the type of a field does not change the type of the struct or enum, but
/// adding/removing fields or changing a fields name or visibility does.
const BASE_STRUCT: &[&str] =
    &[label_strs::generics_of, label_strs::predicates_of, label_strs::type_of];

/// Trait definition `DepNode`s.
/// Extra `DepNode`s for functions and methods.
const EXTRA_ASSOCIATED: &[&str] = &[label_strs::associated_item];

const EXTRA_TRAIT: &[&str] = &[label_strs::trait_of_item];

// Fully Built Labels

const LABELS_CONST: &[&[&str]] = &[BASE_HIR, BASE_CONST];

/// Constant/Typedef in an impl
const LABELS_CONST_IN_IMPL: &[&[&str]] = &[BASE_HIR, BASE_CONST, EXTRA_ASSOCIATED];

/// Trait-Const/Typedef DepNodes
const LABELS_CONST_IN_TRAIT: &[&[&str]] = &[BASE_HIR, BASE_CONST, EXTRA_ASSOCIATED, EXTRA_TRAIT];

/// Function `DepNode`s.
const LABELS_FN: &[&[&str]] = &[BASE_HIR, BASE_MIR, BASE_FN];

/// Method `DepNode`s.
const LABELS_FN_IN_IMPL: &[&[&str]] = &[BASE_HIR, BASE_MIR, BASE_FN, EXTRA_ASSOCIATED];

/// Trait method `DepNode`s.
const LABELS_FN_IN_TRAIT: &[&[&str]] =
    &[BASE_HIR, BASE_MIR, BASE_FN, EXTRA_ASSOCIATED, EXTRA_TRAIT];

/// For generic cases like inline-assembly, modules, etc.
const LABELS_HIR_ONLY: &[&[&str]] = &[BASE_HIR];

/// Impl `DepNode`s.
const LABELS_TRAIT: &[&[&str]] = &[
    BASE_HIR,
    &[label_strs::associated_item_def_ids, label_strs::predicates_of, label_strs::generics_of],
];

/// Impl `DepNode`s.
const LABELS_IMPL: &[&[&str]] = &[BASE_HIR, BASE_IMPL];

/// Abstract data type (struct, enum, union) `DepNode`s.
const LABELS_ADT: &[&[&str]] = &[BASE_HIR, BASE_STRUCT];

// FIXME: Struct/Enum/Unions Fields (there is currently no way to attach these)
//
// Fields are kind of separate from their containers, as they can change independently from
// them. We should at least check
//
//     type_of for these.

type Labels = FxHashSet<String>;

/// Represents the requested configuration by rustc_clean/dirty
struct Assertion {
    clean: Labels,
    dirty: Labels,
}

pub fn check_dirty_clean_annotations(tcx: TyCtxt<'_>) {
    if !tcx.sess.opts.debugging_opts.query_dep_graph {
        return;
    }

    // can't add `#[rustc_clean]` etc without opting in to this feature
    if !tcx.features().rustc_attrs {
        return;
    }

    tcx.dep_graph.with_ignore(|| {
        let krate = tcx.hir().krate();
        let mut dirty_clean_visitor = DirtyCleanVisitor { tcx, checked_attrs: Default::default() };
        krate.visit_all_item_likes(&mut dirty_clean_visitor);

        let mut all_attrs = FindAllAttrs { tcx, found_attrs: vec![] };
        tcx.hir().walk_attributes(&mut all_attrs);

        // Note that we cannot use the existing "unused attribute"-infrastructure
        // here, since that is running before codegen. This is also the reason why
        // all codegen-specific attributes are `AssumedUsed` in rustc_ast::feature_gate.
        all_attrs.report_unchecked_attrs(dirty_clean_visitor.checked_attrs);
    })
}

pub struct DirtyCleanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    checked_attrs: FxHashSet<ast::AttrId>,
}

impl DirtyCleanVisitor<'tcx> {
    /// Possibly "deserialize" the attribute into a clean/dirty assertion
    fn assertion_maybe(&mut self, item_id: LocalDefId, attr: &Attribute) -> Option<Assertion> {
        if !attr.has_name(sym::rustc_clean) {
            // skip: not rustc_clean/dirty
            return None;
        }
        if !check_config(self.tcx, attr) {
            // skip: not the correct `cfg=`
            return None;
        }
        let assertion = self.assertion_auto(item_id, attr);
        Some(assertion)
    }

    /// Gets the "auto" assertion on pre-validated attr, along with the `except` labels.
    fn assertion_auto(&mut self, item_id: LocalDefId, attr: &Attribute) -> Assertion {
        let (name, mut auto) = self.auto_labels(item_id, attr);
        let except = self.except(attr);
        for e in except.iter() {
            if !auto.remove(e) {
                let msg = format!(
                    "`except` specified DepNodes that can not be affected for \"{}\": \"{}\"",
                    name, e
                );
                self.tcx.sess.span_fatal(attr.span, &msg);
            }
        }
        Assertion { clean: auto, dirty: except }
    }

    /// `except=` attribute value
    fn except(&self, attr: &Attribute) -> Labels {
        for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
            if item.has_name(EXCEPT) {
                let value = expect_associated_value(self.tcx, &item);
                return self.resolve_labels(&item, value);
            }
        }
        // if no `label` or `except` is given, only the node's group are asserted
        Labels::default()
    }

    /// Return all DepNode labels that should be asserted for this item.
    /// index=0 is the "name" used for error messages
    fn auto_labels(&mut self, item_id: LocalDefId, attr: &Attribute) -> (&'static str, Labels) {
        let hir_id = self.tcx.hir().local_def_id_to_hir_id(item_id);
        let node = self.tcx.hir().get(hir_id);
        let (name, labels) = match node {
            HirNode::Item(item) => {
                match item.kind {
                    // note: these are in the same order as hir::Item_;
                    // FIXME(michaelwoerister): do commented out ones

                    // // An `extern crate` item, with optional original crate name,
                    // HirItem::ExternCrate(..),  // intentionally no assertions

                    // // `use foo::bar::*;` or `use foo::bar::baz as quux;`
                    // HirItem::Use(..),  // intentionally no assertions

                    // A `static` item
                    HirItem::Static(..) => ("ItemStatic", LABELS_CONST),

                    // A `const` item
                    HirItem::Const(..) => ("ItemConst", LABELS_CONST),

                    // A function declaration
                    HirItem::Fn(..) => ("ItemFn", LABELS_FN),

                    // // A module
                    HirItem::Mod(..) => ("ItemMod", LABELS_HIR_ONLY),

                    // // An external module
                    HirItem::ForeignMod { .. } => ("ItemForeignMod", LABELS_HIR_ONLY),

                    // Module-level inline assembly (from global_asm!)
                    HirItem::GlobalAsm(..) => ("ItemGlobalAsm", LABELS_HIR_ONLY),

                    // A type alias, e.g., `type Foo = Bar<u8>`
                    HirItem::TyAlias(..) => ("ItemTy", LABELS_HIR_ONLY),

                    // An enum definition, e.g., `enum Foo<A, B> {C<A>, D<B>}`
                    HirItem::Enum(..) => ("ItemEnum", LABELS_ADT),

                    // A struct definition, e.g., `struct Foo<A> {x: A}`
                    HirItem::Struct(..) => ("ItemStruct", LABELS_ADT),

                    // A union definition, e.g., `union Foo<A, B> {x: A, y: B}`
                    HirItem::Union(..) => ("ItemUnion", LABELS_ADT),

                    // Represents a Trait Declaration
                    HirItem::Trait(..) => ("ItemTrait", LABELS_TRAIT),

                    // An implementation, eg `impl<A> Trait for Foo { .. }`
                    HirItem::Impl { .. } => ("ItemKind::Impl", LABELS_IMPL),

                    _ => self.tcx.sess.span_fatal(
                        attr.span,
                        &format!(
                            "clean/dirty auto-assertions not yet defined \
                             for Node::Item.node={:?}",
                            item.kind
                        ),
                    ),
                }
            }
            HirNode::TraitItem(item) => match item.kind {
                TraitItemKind::Fn(..) => ("Node::TraitItem", LABELS_FN_IN_TRAIT),
                TraitItemKind::Const(..) => ("NodeTraitConst", LABELS_CONST_IN_TRAIT),
                TraitItemKind::Type(..) => ("NodeTraitType", LABELS_CONST_IN_TRAIT),
            },
            HirNode::ImplItem(item) => match item.kind {
                ImplItemKind::Fn(..) => ("Node::ImplItem", LABELS_FN_IN_IMPL),
                ImplItemKind::Const(..) => ("NodeImplConst", LABELS_CONST_IN_IMPL),
                ImplItemKind::TyAlias(..) => ("NodeImplType", LABELS_CONST_IN_IMPL),
            },
            _ => self.tcx.sess.span_fatal(
                attr.span,
                &format!("clean/dirty auto-assertions not yet defined for {:?}", node),
            ),
        };
        let labels =
            Labels::from_iter(labels.iter().flat_map(|s| s.iter().map(|l| (*l).to_string())));
        (name, labels)
    }

    fn resolve_labels(&self, item: &NestedMetaItem, value: Symbol) -> Labels {
        let mut out = Labels::default();
        for label in value.as_str().split(',') {
            let label = label.trim();
            if DepNode::has_label_string(label) {
                if out.contains(label) {
                    self.tcx.sess.span_fatal(
                        item.span(),
                        &format!("dep-node label `{}` is repeated", label),
                    );
                }
                out.insert(label.to_string());
            } else {
                self.tcx
                    .sess
                    .span_fatal(item.span(), &format!("dep-node label `{}` not recognized", label));
            }
        }
        out
    }

    fn dep_nodes<'l>(
        &self,
        labels: &'l Labels,
        def_id: DefId,
    ) -> impl Iterator<Item = DepNode> + 'l {
        let def_path_hash = self.tcx.def_path_hash(def_id);
        labels.iter().map(move |label| match DepNode::from_label_string(label, def_path_hash) {
            Ok(dep_node) => dep_node,
            Err(()) => unreachable!("label: {}", label),
        })
    }

    fn dep_node_str(&self, dep_node: &DepNode) -> String {
        if let Some(def_id) = dep_node.extract_def_id(self.tcx) {
            format!("{:?}({})", dep_node.kind, self.tcx.def_path_str(def_id))
        } else {
            format!("{:?}({:?})", dep_node.kind, dep_node.hash)
        }
    }

    fn assert_dirty(&self, item_span: Span, dep_node: DepNode) {
        debug!("assert_dirty({:?})", dep_node);

        if self.tcx.dep_graph.is_green(&dep_node) {
            let dep_node_str = self.dep_node_str(&dep_node);
            self.tcx
                .sess
                .span_err(item_span, &format!("`{}` should be dirty but is not", dep_node_str));
        }
    }

    fn assert_clean(&self, item_span: Span, dep_node: DepNode) {
        debug!("assert_clean({:?})", dep_node);

        if self.tcx.dep_graph.is_red(&dep_node) {
            let dep_node_str = self.dep_node_str(&dep_node);
            self.tcx
                .sess
                .span_err(item_span, &format!("`{}` should be clean but is not", dep_node_str));
        }
    }

    fn check_item(&mut self, item_id: LocalDefId, item_span: Span) {
        for attr in self.tcx.get_attrs(item_id.to_def_id()).iter() {
            let assertion = match self.assertion_maybe(item_id, attr) {
                Some(a) => a,
                None => continue,
            };
            self.checked_attrs.insert(attr.id);
            for dep_node in self.dep_nodes(&assertion.clean, item_id.to_def_id()) {
                self.assert_clean(item_span, dep_node);
            }
            for dep_node in self.dep_nodes(&assertion.dirty, item_id.to_def_id()) {
                self.assert_dirty(item_span, dep_node);
            }
        }
    }
}

impl ItemLikeVisitor<'tcx> for DirtyCleanVisitor<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        self.check_item(item.def_id, item.span);
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem<'_>) {
        self.check_item(item.def_id, item.span);
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem<'_>) {
        self.check_item(item.def_id, item.span);
    }

    fn visit_foreign_item(&mut self, item: &hir::ForeignItem<'_>) {
        self.check_item(item.def_id, item.span);
    }
}

/// Given a `#[rustc_clean]` attribute, scan for a `cfg="foo"` attribute and check whether we have
/// a cfg flag called `foo`.
fn check_config(tcx: TyCtxt<'_>, attr: &Attribute) -> bool {
    debug!("check_config(attr={:?})", attr);
    let config = &tcx.sess.parse_sess.config;
    debug!("check_config: config={:?}", config);
    let mut cfg = None;
    for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
        if item.has_name(CFG) {
            let value = expect_associated_value(tcx, &item);
            debug!("check_config: searching for cfg {:?}", value);
            cfg = Some(config.contains(&(value, None)));
        } else if !item.has_name(EXCEPT) {
            tcx.sess.span_err(attr.span, &format!("unknown item `{}`", item.name_or_empty()));
        }
    }

    match cfg {
        None => tcx.sess.span_fatal(attr.span, "no cfg attribute"),
        Some(c) => c,
    }
}

fn expect_associated_value(tcx: TyCtxt<'_>, item: &NestedMetaItem) -> Symbol {
    if let Some(value) = item.value_str() {
        value
    } else {
        let msg = if let Some(ident) = item.ident() {
            format!("associated value expected for `{}`", ident)
        } else {
            "expected an associated value".to_string()
        };

        tcx.sess.span_fatal(item.span(), &msg);
    }
}

// A visitor that collects all #[rustc_clean] attributes from
// the HIR. It is used to verify that we really ran checks for all annotated
// nodes.
pub struct FindAllAttrs<'tcx> {
    tcx: TyCtxt<'tcx>,
    found_attrs: Vec<&'tcx Attribute>,
}

impl FindAllAttrs<'tcx> {
    fn is_active_attr(&mut self, attr: &Attribute) -> bool {
        if attr.has_name(sym::rustc_clean) && check_config(self.tcx, attr) {
            return true;
        }

        false
    }

    fn report_unchecked_attrs(&self, mut checked_attrs: FxHashSet<ast::AttrId>) {
        for attr in &self.found_attrs {
            if !checked_attrs.contains(&attr.id) {
                self.tcx.sess.span_err(attr.span, "found unchecked `#[rustc_clean]` attribute");
                checked_attrs.insert(attr.id);
            }
        }
    }
}

impl intravisit::Visitor<'tcx> for FindAllAttrs<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_attribute(&mut self, _: hir::HirId, attr: &'tcx Attribute) {
        if self.is_active_attr(attr) {
            self.found_attrs.push(attr);
        }
    }
}
