//! Debugging code to test fingerprints computed for query results. For each node marked with
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
//! - `#[rustc_clean(cfg="rev2", loaded_from_disk='typeck")]` asserts that
//!   the query result for `DepNode::typeck(X)` was actually
//!   loaded from disk (not just marked green). This can be useful
//!   to ensure that a test is actually exercising the deserialization
//!   logic for a particular query result. This can be combined with
//!   `except`
//!
//! Errors are reported if we are in the suitable configuration but
//! the required condition is not met.

use rustc_ast::{self as ast, Attribute, MetaItemInner};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{ImplItemKind, ItemKind as HirItem, Node as HirNode, TraitItemKind, intravisit};
use rustc_middle::dep_graph::{DepNode, DepNodeExt, label_strs};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_span::symbol::{Symbol, sym};
use thin_vec::ThinVec;
use tracing::debug;

use crate::errors;

const LOADED_FROM_DISK: Symbol = sym::loaded_from_disk;
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
    // opt_hir_owner_nodes should be computed for all nodes
    label_strs::opt_hir_owner_nodes,
];

/// `impl` implementation of struct/trait
const BASE_IMPL: &[&str] =
    &[label_strs::associated_item_def_ids, label_strs::generics_of, label_strs::impl_trait_header];

/// DepNodes for exported mir bodies, which is relevant in "executable"
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

const EXTRA_TRAIT: &[&str] = &[];

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
const LABELS_TRAIT: &[&[&str]] = &[BASE_HIR, &[
    label_strs::associated_item_def_ids,
    label_strs::predicates_of,
    label_strs::generics_of,
]];

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

type Labels = UnordSet<String>;

/// Represents the requested configuration by rustc_clean/dirty
struct Assertion {
    clean: Labels,
    dirty: Labels,
    loaded_from_disk: Labels,
}

pub(crate) fn check_dirty_clean_annotations(tcx: TyCtxt<'_>) {
    if !tcx.sess.opts.unstable_opts.query_dep_graph {
        return;
    }

    // can't add `#[rustc_clean]` etc without opting into this feature
    if !tcx.features().rustc_attrs() {
        return;
    }

    tcx.dep_graph.with_ignore(|| {
        let mut dirty_clean_visitor = DirtyCleanVisitor { tcx, checked_attrs: Default::default() };

        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.free_items() {
            dirty_clean_visitor.check_item(id.owner_id.def_id);
        }

        for id in crate_items.trait_items() {
            dirty_clean_visitor.check_item(id.owner_id.def_id);
        }

        for id in crate_items.impl_items() {
            dirty_clean_visitor.check_item(id.owner_id.def_id);
        }

        for id in crate_items.foreign_items() {
            dirty_clean_visitor.check_item(id.owner_id.def_id);
        }

        let mut all_attrs = FindAllAttrs { tcx, found_attrs: vec![] };
        tcx.hir().walk_attributes(&mut all_attrs);

        // Note that we cannot use the existing "unused attribute"-infrastructure
        // here, since that is running before codegen. This is also the reason why
        // all codegen-specific attributes are `AssumedUsed` in rustc_ast::feature_gate.
        all_attrs.report_unchecked_attrs(dirty_clean_visitor.checked_attrs);
    })
}

struct DirtyCleanVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    checked_attrs: FxHashSet<ast::AttrId>,
}

impl<'tcx> DirtyCleanVisitor<'tcx> {
    /// Possibly "deserialize" the attribute into a clean/dirty assertion
    fn assertion_maybe(&mut self, item_id: LocalDefId, attr: &Attribute) -> Option<Assertion> {
        assert!(attr.has_name(sym::rustc_clean));
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
        let loaded_from_disk = self.loaded_from_disk(attr);
        for e in except.items().into_sorted_stable_ord() {
            if !auto.remove(e) {
                self.tcx.dcx().emit_fatal(errors::AssertionAuto { span: attr.span, name, e });
            }
        }
        Assertion { clean: auto, dirty: except, loaded_from_disk }
    }

    /// `loaded_from_disk=` attribute value
    fn loaded_from_disk(&self, attr: &Attribute) -> Labels {
        for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
            if item.has_name(LOADED_FROM_DISK) {
                let value = expect_associated_value(self.tcx, &item);
                return self.resolve_labels(&item, value);
            }
        }
        // If `loaded_from_disk=` is not specified, don't assert anything
        Labels::default()
    }

    /// `except=` attribute value
    fn except(&self, attr: &Attribute) -> Labels {
        for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
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
        let node = self.tcx.hir_node_by_def_id(item_id);
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

                    _ => self.tcx.dcx().emit_fatal(errors::UndefinedCleanDirtyItem {
                        span: attr.span,
                        kind: format!("{:?}", item.kind),
                    }),
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
                ImplItemKind::Type(..) => ("NodeImplType", LABELS_CONST_IN_IMPL),
            },
            _ => self.tcx.dcx().emit_fatal(errors::UndefinedCleanDirty {
                span: attr.span,
                kind: format!("{node:?}"),
            }),
        };
        let labels =
            Labels::from_iter(labels.iter().flat_map(|s| s.iter().map(|l| (*l).to_string())));
        (name, labels)
    }

    fn resolve_labels(&self, item: &MetaItemInner, value: Symbol) -> Labels {
        let mut out = Labels::default();
        for label in value.as_str().split(',') {
            let label = label.trim();
            if DepNode::has_label_string(label) {
                if out.contains(label) {
                    self.tcx
                        .dcx()
                        .emit_fatal(errors::RepeatedDepNodeLabel { span: item.span(), label });
                }
                out.insert(label.to_string());
            } else {
                self.tcx
                    .dcx()
                    .emit_fatal(errors::UnrecognizedDepNodeLabel { span: item.span(), label });
            }
        }
        out
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
                .dcx()
                .emit_err(errors::NotDirty { span: item_span, dep_node_str: &dep_node_str });
        }
    }

    fn assert_clean(&self, item_span: Span, dep_node: DepNode) {
        debug!("assert_clean({:?})", dep_node);

        if self.tcx.dep_graph.is_red(&dep_node) {
            let dep_node_str = self.dep_node_str(&dep_node);
            self.tcx
                .dcx()
                .emit_err(errors::NotClean { span: item_span, dep_node_str: &dep_node_str });
        }
    }

    fn assert_loaded_from_disk(&self, item_span: Span, dep_node: DepNode) {
        debug!("assert_loaded_from_disk({:?})", dep_node);

        if !self.tcx.dep_graph.debug_was_loaded_from_disk(dep_node) {
            let dep_node_str = self.dep_node_str(&dep_node);
            self.tcx
                .dcx()
                .emit_err(errors::NotLoaded { span: item_span, dep_node_str: &dep_node_str });
        }
    }

    fn check_item(&mut self, item_id: LocalDefId) {
        let item_span = self.tcx.def_span(item_id.to_def_id());
        let def_path_hash = self.tcx.def_path_hash(item_id.to_def_id());
        for attr in self.tcx.get_attrs(item_id, sym::rustc_clean) {
            let Some(assertion) = self.assertion_maybe(item_id, attr) else {
                continue;
            };
            self.checked_attrs.insert(attr.id);
            for label in assertion.clean.items().into_sorted_stable_ord() {
                let dep_node = DepNode::from_label_string(self.tcx, label, def_path_hash).unwrap();
                self.assert_clean(item_span, dep_node);
            }
            for label in assertion.dirty.items().into_sorted_stable_ord() {
                let dep_node = DepNode::from_label_string(self.tcx, label, def_path_hash).unwrap();
                self.assert_dirty(item_span, dep_node);
            }
            for label in assertion.loaded_from_disk.items().into_sorted_stable_ord() {
                let dep_node = DepNode::from_label_string(self.tcx, label, def_path_hash).unwrap();
                self.assert_loaded_from_disk(item_span, dep_node);
            }
        }
    }
}

/// Given a `#[rustc_clean]` attribute, scan for a `cfg="foo"` attribute and check whether we have
/// a cfg flag called `foo`.
fn check_config(tcx: TyCtxt<'_>, attr: &Attribute) -> bool {
    debug!("check_config(attr={:?})", attr);
    let config = &tcx.sess.psess.config;
    debug!("check_config: config={:?}", config);
    let mut cfg = None;
    for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
        if item.has_name(CFG) {
            let value = expect_associated_value(tcx, &item);
            debug!("check_config: searching for cfg {:?}", value);
            cfg = Some(config.contains(&(value, None)));
        } else if !(item.has_name(EXCEPT) || item.has_name(LOADED_FROM_DISK)) {
            tcx.dcx().emit_err(errors::UnknownItem { span: attr.span, name: item.name_or_empty() });
        }
    }

    match cfg {
        None => tcx.dcx().emit_fatal(errors::NoCfg { span: attr.span }),
        Some(c) => c,
    }
}

fn expect_associated_value(tcx: TyCtxt<'_>, item: &MetaItemInner) -> Symbol {
    if let Some(value) = item.value_str() {
        value
    } else if let Some(ident) = item.ident() {
        tcx.dcx().emit_fatal(errors::AssociatedValueExpectedFor { span: item.span(), ident });
    } else {
        tcx.dcx().emit_fatal(errors::AssociatedValueExpected { span: item.span() });
    }
}

/// A visitor that collects all `#[rustc_clean]` attributes from
/// the HIR. It is used to verify that we really ran checks for all annotated
/// nodes.
struct FindAllAttrs<'tcx> {
    tcx: TyCtxt<'tcx>,
    found_attrs: Vec<&'tcx Attribute>,
}

impl<'tcx> FindAllAttrs<'tcx> {
    fn is_active_attr(&mut self, attr: &Attribute) -> bool {
        if attr.has_name(sym::rustc_clean) && check_config(self.tcx, attr) {
            return true;
        }

        false
    }

    fn report_unchecked_attrs(&self, mut checked_attrs: FxHashSet<ast::AttrId>) {
        for attr in &self.found_attrs {
            if !checked_attrs.contains(&attr.id) {
                self.tcx.dcx().emit_err(errors::UncheckedClean { span: attr.span });
                checked_attrs.insert(attr.id);
            }
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for FindAllAttrs<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_attribute(&mut self, attr: &'tcx Attribute) {
        if self.is_active_attr(attr) {
            self.found_attrs.push(attr);
        }
    }
}
