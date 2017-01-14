// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints in the Rust compiler.
//!
//! This contains lints which can feasibly be implemented as their own
//! AST visitor. Also see `rustc::lint::builtin`, which contains the
//! definitions of lints that are emitted directly inside the main
//! compiler.
//!
//! To add a new lint to rustc, declare it here using `declare_lint!()`.
//! Then add code to emit the new lint in the appropriate circumstances.
//! You can do that in an existing `LintPass` if it makes sense, or in a
//! new `LintPass`, or using `Session::add_lint` elsewhere in the
//! compiler. Only do the latter if the check can't be written cleanly as a
//! `LintPass` (also, note that such lints will need to be defined in
//! `rustc::lint::builtin`, not here).
//!
//! If you define a new `LintPass`, you will also need to add it to the
//! `add_builtin!` or `add_builtin_with_new!` invocation in `lib.rs`.
//! Use the former for unit-like structs and the latter for structs with
//! a `pub fn new()`.

use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::cfg;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::traits::{self, Reveal};
use rustc::hir::map as hir_map;
use util::nodemap::NodeSet;
use lint::{Level, LateContext, LintContext, LintArray};
use lint::{LintPass, LateLintPass, EarlyLintPass, EarlyContext};

use std::collections::HashSet;

use syntax::ast;
use syntax::attr;
use syntax::feature_gate::{AttributeGate, AttributeType, Stability, deprecated_attributes};
use syntax_pos::Span;

use rustc::hir::{self, PatKind};
use rustc::hir::intravisit::FnKind;

use bad_style::{MethodLateContext, method_context};

// hardwired lints from librustc
pub use lint::builtin::*;

declare_lint! {
    WHILE_TRUE,
    Warn,
    "suggest using `loop { }` instead of `while true { }`"
}

#[derive(Copy, Clone)]
pub struct WhileTrue;

impl LintPass for WhileTrue {
    fn get_lints(&self) -> LintArray {
        lint_array!(WHILE_TRUE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for WhileTrue {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        if let hir::ExprWhile(ref cond, ..) = e.node {
            if let hir::ExprLit(ref lit) = cond.node {
                if let ast::LitKind::Bool(true) = lit.node {
                    cx.span_lint(WHILE_TRUE,
                                 e.span,
                                 "denote infinite loops with loop { ... }");
                }
            }
        }
    }
}

declare_lint! {
    BOX_POINTERS,
    Allow,
    "use of owned (Box type) heap memory"
}

#[derive(Copy, Clone)]
pub struct BoxPointers;

impl BoxPointers {
    fn check_heap_type<'a, 'tcx>(&self, cx: &LateContext, span: Span, ty: Ty) {
        for leaf_ty in ty.walk() {
            if let ty::TyBox(_) = leaf_ty.sty {
                let m = format!("type uses owned (Box type) pointers: {}", ty);
                cx.span_lint(BOX_POINTERS, span, &m);
            }
        }
    }
}

impl LintPass for BoxPointers {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_POINTERS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BoxPointers {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match it.node {
            hir::ItemFn(..) |
            hir::ItemTy(..) |
            hir::ItemEnum(..) |
            hir::ItemStruct(..) |
            hir::ItemUnion(..) => {
                let def_id = cx.tcx.map.local_def_id(it.id);
                self.check_heap_type(cx, it.span, cx.tcx.item_type(def_id))
            }
            _ => ()
        }

        // If it's a struct, we also have to check the fields' types
        match it.node {
            hir::ItemStruct(ref struct_def, _) |
            hir::ItemUnion(ref struct_def, _) => {
                for struct_field in struct_def.fields() {
                    let def_id = cx.tcx.map.local_def_id(struct_field.id);
                    self.check_heap_type(cx, struct_field.span,
                                         cx.tcx.item_type(def_id));
                }
            }
            _ => (),
        }
    }

    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        let ty = cx.tables.node_id_to_type(e.id);
        self.check_heap_type(cx, e.span, ty);
    }
}

declare_lint! {
    NON_SHORTHAND_FIELD_PATTERNS,
    Warn,
    "using `Struct { x: x }` instead of `Struct { x }`"
}

#[derive(Copy, Clone)]
pub struct NonShorthandFieldPatterns;

impl LintPass for NonShorthandFieldPatterns {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_SHORTHAND_FIELD_PATTERNS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonShorthandFieldPatterns {
    fn check_pat(&mut self, cx: &LateContext, pat: &hir::Pat) {
        if let PatKind::Struct(_, ref field_pats, _) = pat.node {
            for fieldpat in field_pats {
                if fieldpat.node.is_shorthand {
                    continue;
                }
                if let PatKind::Binding(_, _, ident, None) = fieldpat.node.pat.node {
                    if ident.node == fieldpat.node.name {
                        cx.span_lint(NON_SHORTHAND_FIELD_PATTERNS,
                                     fieldpat.span,
                                     &format!("the `{}:` in this pattern is redundant and can \
                                              be removed",
                                              ident.node))
                    }
                }
            }
        }
    }
}

declare_lint! {
    UNSAFE_CODE,
    Allow,
    "usage of `unsafe` code"
}

#[derive(Copy, Clone)]
pub struct UnsafeCode;

impl LintPass for UnsafeCode {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSAFE_CODE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnsafeCode {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        if let hir::ExprBlock(ref blk) = e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            if blk.rules == hir::UnsafeBlock(hir::UserProvided) {
                cx.span_lint(UNSAFE_CODE, blk.span, "usage of an `unsafe` block");
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match it.node {
            hir::ItemTrait(hir::Unsafety::Unsafe, ..) => {
                cx.span_lint(UNSAFE_CODE, it.span, "declaration of an `unsafe` trait")
            }

            hir::ItemImpl(hir::Unsafety::Unsafe, ..) => {
                cx.span_lint(UNSAFE_CODE, it.span, "implementation of an `unsafe` trait")
            }

            _ => return,
        }
    }

    fn check_fn(&mut self,
                cx: &LateContext,
                fk: FnKind<'tcx>,
                _: &hir::FnDecl,
                _: &hir::Body,
                span: Span,
                _: ast::NodeId) {
        match fk {
            FnKind::ItemFn(_, _, hir::Unsafety::Unsafe, ..) => {
                cx.span_lint(UNSAFE_CODE, span, "declaration of an `unsafe` function")
            }

            FnKind::Method(_, sig, ..) => {
                if sig.unsafety == hir::Unsafety::Unsafe {
                    cx.span_lint(UNSAFE_CODE, span, "implementation of an `unsafe` method")
                }
            }

            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Required(_)) = item.node {
            if sig.unsafety == hir::Unsafety::Unsafe {
                cx.span_lint(UNSAFE_CODE,
                             item.span,
                             "declaration of an `unsafe` method")
            }
        }
    }
}

declare_lint! {
    MISSING_DOCS,
    Allow,
    "detects missing documentation for public members"
}

pub struct MissingDoc {
    /// Stack of IDs of struct definitions.
    struct_def_stack: Vec<ast::NodeId>,

    /// True if inside variant definition
    in_variant: bool,

    /// Stack of whether #[doc(hidden)] is set
    /// at each level which has lint attributes.
    doc_hidden_stack: Vec<bool>,

    /// Private traits or trait items that leaked through. Don't check their methods.
    private_traits: HashSet<ast::NodeId>,
}

impl MissingDoc {
    pub fn new() -> MissingDoc {
        MissingDoc {
            struct_def_stack: vec![],
            in_variant: false,
            doc_hidden_stack: vec![false],
            private_traits: HashSet::new(),
        }
    }

    fn doc_hidden(&self) -> bool {
        *self.doc_hidden_stack.last().expect("empty doc_hidden_stack")
    }

    fn check_missing_docs_attrs(&self,
                                cx: &LateContext,
                                id: Option<ast::NodeId>,
                                attrs: &[ast::Attribute],
                                sp: Span,
                                desc: &'static str) {
        // If we're building a test harness, then warning about
        // documentation is probably not really relevant right now.
        if cx.sess().opts.test {
            return;
        }

        // `#[doc(hidden)]` disables missing_docs check.
        if self.doc_hidden() {
            return;
        }

        // Only check publicly-visible items, using the result from the privacy pass.
        // It's an option so the crate root can also use this function (it doesn't
        // have a NodeId).
        if let Some(id) = id {
            if !cx.access_levels.is_exported(id) {
                return;
            }
        }

        let has_doc = attrs.iter().any(|a| a.is_value_str() && a.name() == "doc");
        if !has_doc {
            cx.span_lint(MISSING_DOCS,
                         sp,
                         &format!("missing documentation for {}", desc));
        }
    }
}

impl LintPass for MissingDoc {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_DOCS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingDoc {
    fn enter_lint_attrs(&mut self, _: &LateContext, attrs: &[ast::Attribute]) {
        let doc_hidden = self.doc_hidden() ||
                         attrs.iter().any(|attr| {
            attr.check_name("doc") &&
            match attr.meta_item_list() {
                None => false,
                Some(l) => attr::list_contains_name(&l[..], "hidden"),
            }
        });
        self.doc_hidden_stack.push(doc_hidden);
    }

    fn exit_lint_attrs(&mut self, _: &LateContext, _attrs: &[ast::Attribute]) {
        self.doc_hidden_stack.pop().expect("empty doc_hidden_stack");
    }

    fn check_struct_def(&mut self,
                        _: &LateContext,
                        _: &hir::VariantData,
                        _: ast::Name,
                        _: &hir::Generics,
                        item_id: ast::NodeId) {
        self.struct_def_stack.push(item_id);
    }

    fn check_struct_def_post(&mut self,
                             _: &LateContext,
                             _: &hir::VariantData,
                             _: ast::Name,
                             _: &hir::Generics,
                             item_id: ast::NodeId) {
        let popped = self.struct_def_stack.pop().expect("empty struct_def_stack");
        assert!(popped == item_id);
    }

    fn check_crate(&mut self, cx: &LateContext, krate: &hir::Crate) {
        self.check_missing_docs_attrs(cx, None, &krate.attrs, krate.span, "crate");
    }

    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        let desc = match it.node {
            hir::ItemFn(..) => "a function",
            hir::ItemMod(..) => "a module",
            hir::ItemEnum(..) => "an enum",
            hir::ItemStruct(..) => "a struct",
            hir::ItemUnion(..) => "a union",
            hir::ItemTrait(.., ref trait_item_refs) => {
                // Issue #11592, traits are always considered exported, even when private.
                if it.vis == hir::Visibility::Inherited {
                    self.private_traits.insert(it.id);
                    for trait_item_ref in trait_item_refs {
                        self.private_traits.insert(trait_item_ref.id.node_id);
                    }
                    return;
                }
                "a trait"
            }
            hir::ItemTy(..) => "a type alias",
            hir::ItemImpl(.., Some(ref trait_ref), _, ref impl_item_refs) => {
                // If the trait is private, add the impl items to private_traits so they don't get
                // reported for missing docs.
                let real_trait = trait_ref.path.def.def_id();
                if let Some(node_id) = cx.tcx.map.as_local_node_id(real_trait) {
                    match cx.tcx.map.find(node_id) {
                        Some(hir_map::NodeItem(item)) => {
                            if item.vis == hir::Visibility::Inherited {
                                for impl_item_ref in impl_item_refs {
                                    self.private_traits.insert(impl_item_ref.id.node_id);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                return;
            }
            hir::ItemConst(..) => "a constant",
            hir::ItemStatic(..) => "a static",
            _ => return,
        };

        self.check_missing_docs_attrs(cx, Some(it.id), &it.attrs, it.span, desc);
    }

    fn check_trait_item(&mut self, cx: &LateContext, trait_item: &hir::TraitItem) {
        if self.private_traits.contains(&trait_item.id) {
            return;
        }

        let desc = match trait_item.node {
            hir::TraitItemKind::Const(..) => "an associated constant",
            hir::TraitItemKind::Method(..) => "a trait method",
            hir::TraitItemKind::Type(..) => "an associated type",
        };

        self.check_missing_docs_attrs(cx,
                                      Some(trait_item.id),
                                      &trait_item.attrs,
                                      trait_item.span,
                                      desc);
    }

    fn check_impl_item(&mut self, cx: &LateContext, impl_item: &hir::ImplItem) {
        // If the method is an impl for a trait, don't doc.
        if method_context(cx, impl_item.id, impl_item.span) == MethodLateContext::TraitImpl {
            return;
        }

        let desc = match impl_item.node {
            hir::ImplItemKind::Const(..) => "an associated constant",
            hir::ImplItemKind::Method(..) => "a method",
            hir::ImplItemKind::Type(_) => "an associated type",
        };
        self.check_missing_docs_attrs(cx,
                                      Some(impl_item.id),
                                      &impl_item.attrs,
                                      impl_item.span,
                                      desc);
    }

    fn check_struct_field(&mut self, cx: &LateContext, sf: &hir::StructField) {
        if !sf.is_positional() {
            if sf.vis == hir::Public || self.in_variant {
                let cur_struct_def = *self.struct_def_stack
                    .last()
                    .expect("empty struct_def_stack");
                self.check_missing_docs_attrs(cx,
                                              Some(cur_struct_def),
                                              &sf.attrs,
                                              sf.span,
                                              "a struct field")
            }
        }
    }

    fn check_variant(&mut self, cx: &LateContext, v: &hir::Variant, _: &hir::Generics) {
        self.check_missing_docs_attrs(cx,
                                      Some(v.node.data.id()),
                                      &v.node.attrs,
                                      v.span,
                                      "a variant");
        assert!(!self.in_variant);
        self.in_variant = true;
    }

    fn check_variant_post(&mut self, _: &LateContext, _: &hir::Variant, _: &hir::Generics) {
        assert!(self.in_variant);
        self.in_variant = false;
    }
}

declare_lint! {
    pub MISSING_COPY_IMPLEMENTATIONS,
    Allow,
    "detects potentially-forgotten implementations of `Copy`"
}

#[derive(Copy, Clone)]
pub struct MissingCopyImplementations;

impl LintPass for MissingCopyImplementations {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_COPY_IMPLEMENTATIONS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingCopyImplementations {
    fn check_item(&mut self, cx: &LateContext, item: &hir::Item) {
        if !cx.access_levels.is_reachable(item.id) {
            return;
        }
        let (def, ty) = match item.node {
            hir::ItemStruct(_, ref ast_generics) => {
                if ast_generics.is_parameterized() {
                    return;
                }
                let def = cx.tcx.lookup_adt_def(cx.tcx.map.local_def_id(item.id));
                (def, cx.tcx.mk_adt(def, cx.tcx.intern_substs(&[])))
            }
            hir::ItemUnion(_, ref ast_generics) => {
                if ast_generics.is_parameterized() {
                    return;
                }
                let def = cx.tcx.lookup_adt_def(cx.tcx.map.local_def_id(item.id));
                (def, cx.tcx.mk_adt(def, cx.tcx.intern_substs(&[])))
            }
            hir::ItemEnum(_, ref ast_generics) => {
                if ast_generics.is_parameterized() {
                    return;
                }
                let def = cx.tcx.lookup_adt_def(cx.tcx.map.local_def_id(item.id));
                (def, cx.tcx.mk_adt(def, cx.tcx.intern_substs(&[])))
            }
            _ => return,
        };
        if def.has_dtor() {
            return;
        }
        let parameter_environment = cx.tcx.empty_parameter_environment();
        // FIXME (@jroesch) should probably inver this so that the parameter env still impls this
        // method
        if !ty.moves_by_default(cx.tcx, &parameter_environment, item.span) {
            return;
        }
        if parameter_environment.can_type_implement_copy(cx.tcx, ty, item.span).is_ok() {
            cx.span_lint(MISSING_COPY_IMPLEMENTATIONS,
                         item.span,
                         "type could implement `Copy`; consider adding `impl \
                          Copy`")
        }
    }
}

declare_lint! {
    MISSING_DEBUG_IMPLEMENTATIONS,
    Allow,
    "detects missing implementations of fmt::Debug"
}

pub struct MissingDebugImplementations {
    impling_types: Option<NodeSet>,
}

impl MissingDebugImplementations {
    pub fn new() -> MissingDebugImplementations {
        MissingDebugImplementations { impling_types: None }
    }
}

impl LintPass for MissingDebugImplementations {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_DEBUG_IMPLEMENTATIONS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingDebugImplementations {
    fn check_item(&mut self, cx: &LateContext, item: &hir::Item) {
        if !cx.access_levels.is_reachable(item.id) {
            return;
        }

        match item.node {
            hir::ItemStruct(..) |
            hir::ItemUnion(..) |
            hir::ItemEnum(..) => {}
            _ => return,
        }

        let debug = match cx.tcx.lang_items.debug_trait() {
            Some(debug) => debug,
            None => return,
        };

        if self.impling_types.is_none() {
            let debug_def = cx.tcx.lookup_trait_def(debug);
            let mut impls = NodeSet();
            debug_def.for_each_impl(cx.tcx, |d| {
                if let Some(ty_def) = cx.tcx.item_type(d).ty_to_def_id() {
                    if let Some(node_id) = cx.tcx.map.as_local_node_id(ty_def) {
                        impls.insert(node_id);
                    }
                }
            });

            self.impling_types = Some(impls);
            debug!("{:?}", self.impling_types);
        }

        if !self.impling_types.as_ref().unwrap().contains(&item.id) {
            cx.span_lint(MISSING_DEBUG_IMPLEMENTATIONS,
                         item.span,
                         "type does not implement `fmt::Debug`; consider adding #[derive(Debug)] \
                          or a manual implementation")
        }
    }
}

declare_lint! {
    DEPRECATED_ATTR,
    Warn,
    "detects use of deprecated attributes"
}

/// Checks for use of attributes which have been deprecated.
#[derive(Clone)]
pub struct DeprecatedAttr {
    // This is not free to compute, so we want to keep it around, rather than
    // compute it for every attribute.
    depr_attrs: Vec<&'static (&'static str, AttributeType, AttributeGate)>,
}

impl DeprecatedAttr {
    pub fn new() -> DeprecatedAttr {
        DeprecatedAttr {
            depr_attrs: deprecated_attributes(),
        }
    }
}

impl LintPass for DeprecatedAttr {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEPRECATED_ATTR)
    }
}

impl EarlyLintPass for DeprecatedAttr {
    fn check_attribute(&mut self, cx: &EarlyContext, attr: &ast::Attribute) {
        let name = attr.name();
        for &&(n, _, ref g) in &self.depr_attrs {
            if name == n {
                if let &AttributeGate::Gated(Stability::Deprecated(link),
                                             ref name,
                                             ref reason,
                                             _) = g {
                    cx.span_lint(DEPRECATED,
                                 attr.span,
                                 &format!("use of deprecated attribute `{}`: {}. See {}",
                                          name, reason, link));
                }
                return;
            }
        }
    }
}

declare_lint! {
    pub UNCONDITIONAL_RECURSION,
    Warn,
    "functions that cannot return without calling themselves"
}

#[derive(Copy, Clone)]
pub struct UnconditionalRecursion;


impl LintPass for UnconditionalRecursion {
    fn get_lints(&self) -> LintArray {
        lint_array![UNCONDITIONAL_RECURSION]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnconditionalRecursion {
    fn check_fn(&mut self,
                cx: &LateContext,
                fn_kind: FnKind,
                _: &hir::FnDecl,
                body: &hir::Body,
                sp: Span,
                id: ast::NodeId) {
        let method = match fn_kind {
            FnKind::ItemFn(..) => None,
            FnKind::Method(..) => {
                Some(cx.tcx.associated_item(cx.tcx.map.local_def_id(id)))
            }
            // closures can't recur, so they don't matter.
            FnKind::Closure(_) => return,
        };

        // Walk through this function (say `f`) looking to see if
        // every possible path references itself, i.e. the function is
        // called recursively unconditionally. This is done by trying
        // to find a path from the entry node to the exit node that
        // *doesn't* call `f` by traversing from the entry while
        // pretending that calls of `f` are sinks (i.e. ignoring any
        // exit edges from them).
        //
        // NB. this has an edge case with non-returning statements,
        // like `loop {}` or `panic!()`: control flow never reaches
        // the exit node through these, so one can have a function
        // that never actually calls itselfs but is still picked up by
        // this lint:
        //
        //     fn f(cond: bool) {
        //         if !cond { panic!() } // could come from `assert!(cond)`
        //         f(false)
        //     }
        //
        // In general, functions of that form may be able to call
        // itself a finite number of times and then diverge. The lint
        // considers this to be an error for two reasons, (a) it is
        // easier to implement, and (b) it seems rare to actually want
        // to have behaviour like the above, rather than
        // e.g. accidentally recurring after an assert.

        let cfg = cfg::CFG::new(cx.tcx, &body.value);

        let mut work_queue = vec![cfg.entry];
        let mut reached_exit_without_self_call = false;
        let mut self_call_spans = vec![];
        let mut visited = HashSet::new();

        while let Some(idx) = work_queue.pop() {
            if idx == cfg.exit {
                // found a path!
                reached_exit_without_self_call = true;
                break;
            }

            let cfg_id = idx.node_id();
            if visited.contains(&cfg_id) {
                // already done
                continue;
            }
            visited.insert(cfg_id);

            let node_id = cfg.graph.node_data(idx).id();

            // is this a recursive call?
            let self_recursive = if node_id != ast::DUMMY_NODE_ID {
                match method {
                    Some(ref method) => expr_refers_to_this_method(cx, method, node_id),
                    None => expr_refers_to_this_fn(cx, id, node_id),
                }
            } else {
                false
            };
            if self_recursive {
                self_call_spans.push(cx.tcx.map.span(node_id));
                // this is a self call, so we shouldn't explore past
                // this node in the CFG.
                continue;
            }
            // add the successors of this node to explore the graph further.
            for (_, edge) in cfg.graph.outgoing_edges(idx) {
                let target_idx = edge.target();
                let target_cfg_id = target_idx.node_id();
                if !visited.contains(&target_cfg_id) {
                    work_queue.push(target_idx)
                }
            }
        }

        // Check the number of self calls because a function that
        // doesn't return (e.g. calls a `-> !` function or `loop { /*
        // no break */ }`) shouldn't be linted unless it actually
        // recurs.
        if !reached_exit_without_self_call && !self_call_spans.is_empty() {
            let mut db = cx.struct_span_lint(UNCONDITIONAL_RECURSION,
                                             sp,
                                             "function cannot return without recurring");

            // FIXME #19668: these could be span_lint_note's instead of this manual guard.
            if cx.current_level(UNCONDITIONAL_RECURSION) != Level::Allow {
                // offer some help to the programmer.
                for call in &self_call_spans {
                    db.span_note(*call, "recursive call site");
                }
                db.help("a `loop` may express intention \
                         better if this is on purpose");
            }
            db.emit();
        }

        // all done
        return;

        // Functions for identifying if the given Expr NodeId `id`
        // represents a call to the function `fn_id`/method `method`.

        fn expr_refers_to_this_fn(cx: &LateContext, fn_id: ast::NodeId, id: ast::NodeId) -> bool {
            match cx.tcx.map.get(id) {
                hir_map::NodeExpr(&hir::Expr { node: hir::ExprCall(ref callee, _), .. }) => {
                    let def = if let hir::ExprPath(ref qpath) = callee.node {
                        cx.tables.qpath_def(qpath, callee.id)
                    } else {
                        return false;
                    };
                    def.def_id() == cx.tcx.map.local_def_id(fn_id)
                }
                _ => false,
            }
        }

        // Check if the expression `id` performs a call to `method`.
        fn expr_refers_to_this_method(cx: &LateContext,
                                      method: &ty::AssociatedItem,
                                      id: ast::NodeId)
                                      -> bool {
            use rustc::ty::adjustment::*;

            // Check for method calls and overloaded operators.
            let opt_m = cx.tables.method_map.get(&ty::MethodCall::expr(id)).cloned();
            if let Some(m) = opt_m {
                if method_call_refers_to_method(cx.tcx, method, m.def_id, m.substs, id) {
                    return true;
                }
            }

            // Check for overloaded autoderef method calls.
            let opt_adj = cx.tables.adjustments.get(&id).cloned();
            if let Some(Adjustment { kind: Adjust::DerefRef { autoderefs, .. }, .. }) = opt_adj {
                for i in 0..autoderefs {
                    let method_call = ty::MethodCall::autoderef(id, i as u32);
                    if let Some(m) = cx.tables.method_map.get(&method_call).cloned() {
                        if method_call_refers_to_method(cx.tcx, method, m.def_id, m.substs, id) {
                            return true;
                        }
                    }
                }
            }

            // Check for calls to methods via explicit paths (e.g. `T::method()`).
            match cx.tcx.map.get(id) {
                hir_map::NodeExpr(&hir::Expr { node: hir::ExprCall(ref callee, _), .. }) => {
                    let def = if let hir::ExprPath(ref qpath) = callee.node {
                        cx.tables.qpath_def(qpath, callee.id)
                    } else {
                        return false;
                    };
                    match def {
                        Def::Method(def_id) => {
                            let substs = cx.tables.node_id_item_substs(callee.id)
                                .unwrap_or_else(|| cx.tcx.intern_substs(&[]));
                            method_call_refers_to_method(
                                cx.tcx, method, def_id, substs, id)
                        }
                        _ => false,
                    }
                }
                _ => false,
            }
        }

        // Check if the method call to the method with the ID `callee_id`
        // and instantiated with `callee_substs` refers to method `method`.
        fn method_call_refers_to_method<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                  method: &ty::AssociatedItem,
                                                  callee_id: DefId,
                                                  callee_substs: &Substs<'tcx>,
                                                  expr_id: ast::NodeId)
                                                  -> bool {
            let callee_item = tcx.associated_item(callee_id);

            match callee_item.container {
                // This is an inherent method, so the `def_id` refers
                // directly to the method definition.
                ty::ImplContainer(_) => callee_id == method.def_id,

                // A trait method, from any number of possible sources.
                // Attempt to select a concrete impl before checking.
                ty::TraitContainer(trait_def_id) => {
                    let trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, callee_substs);
                    let trait_ref = ty::Binder(trait_ref);
                    let span = tcx.map.span(expr_id);
                    let obligation =
                        traits::Obligation::new(traits::ObligationCause::misc(span, expr_id),
                                                trait_ref.to_poly_trait_predicate());

                    // unwrap() is ok here b/c `method` is the method
                    // defined in this crate whose body we are
                    // checking, so it's always local
                    let node_id = tcx.map.as_local_node_id(method.def_id).unwrap();

                    let param_env = ty::ParameterEnvironment::for_item(tcx, node_id);
                    tcx.infer_ctxt(param_env, Reveal::NotSpecializable).enter(|infcx| {
                        let mut selcx = traits::SelectionContext::new(&infcx);
                        match selcx.select(&obligation) {
                            // The method comes from a `T: Trait` bound.
                            // If `T` is `Self`, then this call is inside
                            // a default method definition.
                            Ok(Some(traits::VtableParam(_))) => {
                                let on_self = trait_ref.self_ty().is_self();
                                // We can only be recurring in a default
                                // method if we're being called literally
                                // on the `Self` type.
                                on_self && callee_id == method.def_id
                            }

                            // The `impl` is known, so we check that with a
                            // special case:
                            Ok(Some(traits::VtableImpl(vtable_impl))) => {
                                let container = ty::ImplContainer(vtable_impl.impl_def_id);
                                // It matches if it comes from the same impl,
                                // and has the same method name.
                                container == method.container && callee_item.name == method.name
                            }

                            // There's no way to know if this call is
                            // recursive, so we assume it's not.
                            _ => false,
                        }
                    })
                }
            }
        }
    }
}

declare_lint! {
    PLUGIN_AS_LIBRARY,
    Warn,
    "compiler plugin used as ordinary library in non-plugin crate"
}

#[derive(Copy, Clone)]
pub struct PluginAsLibrary;

impl LintPass for PluginAsLibrary {
    fn get_lints(&self) -> LintArray {
        lint_array![PLUGIN_AS_LIBRARY]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PluginAsLibrary {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        if cx.sess().plugin_registrar_fn.get().is_some() {
            // We're compiling a plugin; it's fine to link other plugins.
            return;
        }

        match it.node {
            hir::ItemExternCrate(..) => (),
            _ => return,
        };

        let prfn = match cx.sess().cstore.extern_mod_stmt_cnum(it.id) {
            Some(cnum) => cx.sess().cstore.plugin_registrar_fn(cnum),
            None => {
                // Probably means we aren't linking the crate for some reason.
                //
                // Not sure if / when this could happen.
                return;
            }
        };

        if prfn.is_some() {
            cx.span_lint(PLUGIN_AS_LIBRARY,
                         it.span,
                         "compiler plugin used as an ordinary library");
        }
    }
}

declare_lint! {
    PRIVATE_NO_MANGLE_FNS,
    Warn,
    "functions marked #[no_mangle] should be exported"
}

declare_lint! {
    PRIVATE_NO_MANGLE_STATICS,
    Warn,
    "statics marked #[no_mangle] should be exported"
}

declare_lint! {
    NO_MANGLE_CONST_ITEMS,
    Deny,
    "const items will not have their symbols exported"
}

declare_lint! {
    NO_MANGLE_GENERIC_ITEMS,
    Warn,
    "generic items must be mangled"
}

#[derive(Copy, Clone)]
pub struct InvalidNoMangleItems;

impl LintPass for InvalidNoMangleItems {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRIVATE_NO_MANGLE_FNS,
                    PRIVATE_NO_MANGLE_STATICS,
                    NO_MANGLE_CONST_ITEMS,
                    NO_MANGLE_GENERIC_ITEMS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for InvalidNoMangleItems {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match it.node {
            hir::ItemFn(.., ref generics, _) => {
                if attr::contains_name(&it.attrs, "no_mangle") {
                    if !cx.access_levels.is_reachable(it.id) {
                        let msg = format!("function {} is marked #[no_mangle], but not exported",
                                          it.name);
                        cx.span_lint(PRIVATE_NO_MANGLE_FNS, it.span, &msg);
                    }
                    if generics.is_parameterized() {
                        cx.span_lint(NO_MANGLE_GENERIC_ITEMS,
                                     it.span,
                                     "generic functions must be mangled");
                    }
                }
            }
            hir::ItemStatic(..) => {
                if attr::contains_name(&it.attrs, "no_mangle") &&
                   !cx.access_levels.is_reachable(it.id) {
                    let msg = format!("static {} is marked #[no_mangle], but not exported",
                                      it.name);
                    cx.span_lint(PRIVATE_NO_MANGLE_STATICS, it.span, &msg);
                }
            }
            hir::ItemConst(..) => {
                if attr::contains_name(&it.attrs, "no_mangle") {
                    // Const items do not refer to a particular location in memory, and therefore
                    // don't have anything to attach a symbol to
                    let msg = "const items should never be #[no_mangle], consider instead using \
                               `pub static`";
                    cx.span_lint(NO_MANGLE_CONST_ITEMS, it.span, msg);
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy)]
pub struct MutableTransmutes;

declare_lint! {
    MUTABLE_TRANSMUTES,
    Deny,
    "mutating transmuted &mut T from &T may cause undefined behavior"
}

impl LintPass for MutableTransmutes {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUTABLE_TRANSMUTES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MutableTransmutes {
    fn check_expr(&mut self, cx: &LateContext, expr: &hir::Expr) {
        use syntax::abi::Abi::RustIntrinsic;

        let msg = "mutating transmuted &mut T from &T may cause undefined behavior, \
                   consider instead using an UnsafeCell";
        match get_transmute_from_to(cx, expr) {
            Some((&ty::TyRef(_, from_mt), &ty::TyRef(_, to_mt))) => {
                if to_mt.mutbl == hir::Mutability::MutMutable &&
                   from_mt.mutbl == hir::Mutability::MutImmutable {
                    cx.span_lint(MUTABLE_TRANSMUTES, expr.span, msg);
                }
            }
            _ => (),
        }

        fn get_transmute_from_to<'a, 'tcx>
            (cx: &LateContext<'a, 'tcx>,
             expr: &hir::Expr)
             -> Option<(&'tcx ty::TypeVariants<'tcx>, &'tcx ty::TypeVariants<'tcx>)> {
            let def = if let hir::ExprPath(ref qpath) = expr.node {
                cx.tables.qpath_def(qpath, expr.id)
            } else {
                return None;
            };
            if let Def::Fn(did) = def {
                if !def_id_is_transmute(cx, did) {
                    return None;
                }
                let typ = cx.tables.node_id_to_type(expr.id);
                match typ.sty {
                    ty::TyFnDef(.., ref bare_fn) if bare_fn.abi == RustIntrinsic => {
                        let from = bare_fn.sig.skip_binder().inputs()[0];
                        let to = bare_fn.sig.skip_binder().output();
                        return Some((&from.sty, &to.sty));
                    }
                    _ => (),
                }
            }
            None
        }

        fn def_id_is_transmute(cx: &LateContext, def_id: DefId) -> bool {
            match cx.tcx.item_type(def_id).sty {
                ty::TyFnDef(.., ref bfty) if bfty.abi == RustIntrinsic => (),
                _ => return false,
            }
            cx.tcx.item_name(def_id) == "transmute"
        }
    }
}

/// Forbids using the `#[feature(...)]` attribute
#[derive(Copy, Clone)]
pub struct UnstableFeatures;

declare_lint! {
    UNSTABLE_FEATURES,
    Allow,
    "enabling unstable features (deprecated. do not use)"
}

impl LintPass for UnstableFeatures {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSTABLE_FEATURES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnstableFeatures {
    fn check_attribute(&mut self, ctx: &LateContext, attr: &ast::Attribute) {
        if attr.meta().check_name("feature") {
            if let Some(items) = attr.meta().meta_item_list() {
                for item in items {
                    ctx.span_lint(UNSTABLE_FEATURES, item.span(), "unstable feature");
                }
            }
        }
    }
}

/// Lint for unions that contain fields with possibly non-trivial destructors.
pub struct UnionsWithDropFields;

declare_lint! {
    UNIONS_WITH_DROP_FIELDS,
    Warn,
    "use of unions that contain fields with possibly non-trivial drop code"
}

impl LintPass for UnionsWithDropFields {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNIONS_WITH_DROP_FIELDS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnionsWithDropFields {
    fn check_item(&mut self, ctx: &LateContext, item: &hir::Item) {
        if let hir::ItemUnion(ref vdata, _) = item.node {
            let param_env = &ty::ParameterEnvironment::for_item(ctx.tcx, item.id);
            for field in vdata.fields() {
                let field_ty = ctx.tcx.item_type(ctx.tcx.map.local_def_id(field.id));
                if ctx.tcx.type_needs_drop_given_env(field_ty, param_env) {
                    ctx.span_lint(UNIONS_WITH_DROP_FIELDS,
                                  field.span,
                                  "union contains a field with possibly non-trivial drop code, \
                                   drop code of union fields is ignored when dropping the union");
                    return;
                }
            }
        }
    }
}
