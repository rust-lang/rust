#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use rustc_ast as ast;
use rustc_ast::util::parser::{self, AssocOp, Fixity};
use rustc_ast_pretty::pp::Breaks::{Consistent, Inconsistent};
use rustc_ast_pretty::pp::{self, Breaks};
use rustc_ast_pretty::pprust::{Comments, PrintState};
use rustc_hir as hir;
use rustc_hir::LifetimeParamKind;
use rustc_hir::{BindingAnnotation, ByRef, GenericArg, GenericParam, GenericParamKind, Node, Term};
use rustc_hir::{GenericBound, PatKind, RangeEnd, TraitBoundModifier};
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, Ident, IdentPrinter, Symbol};
use rustc_span::{self, FileName};
use rustc_target::spec::abi::Abi;

use std::cell::Cell;
use std::vec;

pub fn id_to_string(map: &dyn rustc_hir::intravisit::Map<'_>, hir_id: hir::HirId) -> String {
    to_string(&map, |s| s.print_node(map.find(hir_id).unwrap()))
}

pub enum AnnNode<'a> {
    Name(&'a Symbol),
    Block(&'a hir::Block<'a>),
    Item(&'a hir::Item<'a>),
    SubItem(hir::HirId),
    Expr(&'a hir::Expr<'a>),
    Pat(&'a hir::Pat<'a>),
    Arm(&'a hir::Arm<'a>),
}

pub enum Nested {
    Item(hir::ItemId),
    TraitItem(hir::TraitItemId),
    ImplItem(hir::ImplItemId),
    ForeignItem(hir::ForeignItemId),
    Body(hir::BodyId),
    BodyParamPat(hir::BodyId, usize),
}

pub trait PpAnn {
    fn nested(&self, _state: &mut State<'_>, _nested: Nested) {}
    fn pre(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
    fn post(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
}

pub struct NoAnn;
impl PpAnn for NoAnn {}
pub const NO_ANN: &dyn PpAnn = &NoAnn;

/// Identical to the `PpAnn` implementation for `hir::Crate`,
/// except it avoids creating a dependency on the whole crate.
impl PpAnn for &dyn rustc_hir::intravisit::Map<'_> {
    fn nested(&self, state: &mut State<'_>, nested: Nested) {
        match nested {
            Nested::Item(id) => state.print_item(self.item(id)),
            Nested::TraitItem(id) => state.print_trait_item(self.trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.impl_item(id)),
            Nested::ForeignItem(id) => state.print_foreign_item(self.foreign_item(id)),
            Nested::Body(id) => state.print_expr(&self.body(id).value),
            Nested::BodyParamPat(id, i) => state.print_pat(self.body(id).params[i].pat),
        }
    }
}

pub struct State<'a> {
    pub s: pp::Printer,
    comments: Option<Comments<'a>>,
    attrs: &'a dyn Fn(hir::HirId) -> &'a [ast::Attribute],
    ann: &'a (dyn PpAnn + 'a),
}

impl<'a> State<'a> {
    pub fn print_node(&mut self, node: Node<'_>) {
        match node {
            Node::Param(a) => self.print_param(a),
            Node::Item(a) => self.print_item(a),
            Node::ForeignItem(a) => self.print_foreign_item(a),
            Node::TraitItem(a) => self.print_trait_item(a),
            Node::ImplItem(a) => self.print_impl_item(a),
            Node::Variant(a) => self.print_variant(a),
            Node::AnonConst(a) => self.print_anon_const(a),
            Node::Expr(a) => self.print_expr(a),
            Node::ExprField(a) => self.print_expr_field(&a),
            Node::Stmt(a) => self.print_stmt(a),
            Node::PathSegment(a) => self.print_path_segment(a),
            Node::Ty(a) => self.print_type(a),
            Node::TypeBinding(a) => self.print_type_binding(a),
            Node::TraitRef(a) => self.print_trait_ref(a),
            Node::Pat(a) => self.print_pat(a),
            Node::PatField(a) => self.print_patfield(&a),
            Node::Arm(a) => self.print_arm(a),
            Node::Infer(_) => self.word("_"),
            Node::Block(a) => {
                // Containing cbox, will be closed by print-block at `}`.
                self.cbox(INDENT_UNIT);
                // Head-ibox, will be closed by print-block after `{`.
                self.ibox(0);
                self.print_block(a);
            }
            Node::Lifetime(a) => self.print_lifetime(a),
            Node::GenericParam(_) => panic!("cannot print Node::GenericParam"),
            Node::Field(_) => panic!("cannot print Node::Field"),
            // These cases do not carry enough information in the
            // `hir_map` to reconstruct their full structure for pretty
            // printing.
            Node::Ctor(..) => panic!("cannot print isolated Ctor"),
            Node::Local(a) => self.print_local_decl(a),
            Node::Crate(..) => panic!("cannot print Crate"),
        }
    }
}

impl std::ops::Deref for State<'_> {
    type Target = pp::Printer;
    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

impl std::ops::DerefMut for State<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.s
    }
}

impl<'a> PrintState<'a> for State<'a> {
    fn comments(&mut self) -> &mut Option<Comments<'a>> {
        &mut self.comments
    }

    fn print_ident(&mut self, ident: Ident) {
        self.word(IdentPrinter::for_ast_ident(ident, ident.is_raw_guess()).to_string());
        self.ann.post(self, AnnNode::Name(&ident.name))
    }

    fn print_generic_args(&mut self, _: &ast::GenericArgs, _colons_before_params: bool) {
        panic!("AST generic args printed by HIR pretty-printer");
    }
}

pub const INDENT_UNIT: isize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(
    sm: &'a SourceMap,
    krate: &hir::Mod<'_>,
    filename: FileName,
    input: String,
    attrs: &'a dyn Fn(hir::HirId) -> &'a [ast::Attribute],
    ann: &'a dyn PpAnn,
) -> String {
    let mut s = State::new_from_input(sm, filename, input, attrs, ann);

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    s.print_mod(krate, (*attrs)(hir::CRATE_HIR_ID));
    s.print_remaining_comments();
    s.s.eof()
}

impl<'a> State<'a> {
    pub fn new_from_input(
        sm: &'a SourceMap,
        filename: FileName,
        input: String,
        attrs: &'a dyn Fn(hir::HirId) -> &'a [ast::Attribute],
        ann: &'a dyn PpAnn,
    ) -> State<'a> {
        State {
            s: pp::Printer::new(),
            comments: Some(Comments::new(sm, filename, input)),
            attrs,
            ann,
        }
    }

    fn attrs(&self, id: hir::HirId) -> &'a [ast::Attribute] {
        (self.attrs)(id)
    }
}

pub fn to_string<F>(ann: &dyn PpAnn, f: F) -> String
where
    F: FnOnce(&mut State<'_>),
{
    let mut printer = State { s: pp::Printer::new(), comments: None, attrs: &|_| &[], ann };
    f(&mut printer);
    printer.s.eof()
}

pub fn generic_params_to_string(generic_params: &[GenericParam<'_>]) -> String {
    to_string(NO_ANN, |s| s.print_generic_params(generic_params))
}

pub fn bounds_to_string<'b>(bounds: impl IntoIterator<Item = &'b hir::GenericBound<'b>>) -> String {
    to_string(NO_ANN, |s| s.print_bounds("", bounds))
}

pub fn ty_to_string(ty: &hir::Ty<'_>) -> String {
    to_string(NO_ANN, |s| s.print_type(ty))
}

pub fn path_segment_to_string(segment: &hir::PathSegment<'_>) -> String {
    to_string(NO_ANN, |s| s.print_path_segment(segment))
}

pub fn path_to_string(segment: &hir::Path<'_>) -> String {
    to_string(NO_ANN, |s| s.print_path(segment, false))
}

pub fn qpath_to_string(segment: &hir::QPath<'_>) -> String {
    to_string(NO_ANN, |s| s.print_qpath(segment, false))
}

pub fn fn_to_string(
    decl: &hir::FnDecl<'_>,
    header: hir::FnHeader,
    name: Option<Symbol>,
    generics: &hir::Generics<'_>,
    arg_names: &[Ident],
    body_id: Option<hir::BodyId>,
) -> String {
    to_string(NO_ANN, |s| s.print_fn(decl, header, name, generics, arg_names, body_id))
}

pub fn enum_def_to_string(
    enum_definition: &hir::EnumDef<'_>,
    generics: &hir::Generics<'_>,
    name: Symbol,
    span: rustc_span::Span,
) -> String {
    to_string(NO_ANN, |s| s.print_enum_def(enum_definition, generics, name, span))
}

impl<'a> State<'a> {
    pub fn bclose_maybe_open(&mut self, span: rustc_span::Span, close_box: bool) {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -(INDENT_UNIT as isize));
        self.word("}");
        if close_box {
            self.end(); // close the outer-box
        }
    }

    pub fn bclose(&mut self, span: rustc_span::Span) {
        self.bclose_maybe_open(span, true)
    }

    pub fn commasep_cmnt<T, F, G>(&mut self, b: Breaks, elts: &[T], mut op: F, mut get_span: G)
    where
        F: FnMut(&mut State<'_>, &T),
        G: FnMut(&T) -> rustc_span::Span,
    {
        self.rbox(0, b);
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            self.maybe_print_comment(get_span(elt).hi());
            op(self, elt);
            i += 1;
            if i < len {
                self.word(",");
                self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi()));
                self.space_if_not_bol();
            }
        }
        self.end();
    }

    pub fn commasep_exprs(&mut self, b: Breaks, exprs: &[hir::Expr<'_>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e), |e| e.span);
    }

    pub fn print_mod(&mut self, _mod: &hir::Mod<'_>, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for &item_id in _mod.item_ids {
            self.ann.nested(self, Nested::Item(item_id));
        }
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &hir::Lifetime) {
        if !lifetime.is_elided() {
            self.print_lifetime(lifetime);
            self.nbsp();
        }
    }

    pub fn print_type(&mut self, ty: &hir::Ty<'_>) {
        self.maybe_print_comment(ty.span.lo());
        self.ibox(0);
        match ty.kind {
            hir::TyKind::Slice(ty) => {
                self.word("[");
                self.print_type(ty);
                self.word("]");
            }
            hir::TyKind::Ptr(ref mt) => {
                self.word("*");
                self.print_mt(mt, true);
            }
            hir::TyKind::Ref(ref lifetime, ref mt) => {
                self.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt, false);
            }
            hir::TyKind::Never => {
                self.word("!");
            }
            hir::TyKind::Tup(elts) => {
                self.popen();
                self.commasep(Inconsistent, elts, |s, ty| s.print_type(ty));
                if elts.len() == 1 {
                    self.word(",");
                }
                self.pclose();
            }
            hir::TyKind::BareFn(f) => {
                self.print_ty_fn(f.abi, f.unsafety, f.decl, None, f.generic_params, f.param_names);
            }
            hir::TyKind::OpaqueDef(..) => self.word("/*impl Trait*/"),
            hir::TyKind::Path(ref qpath) => self.print_qpath(qpath, false),
            hir::TyKind::TraitObject(bounds, ref lifetime, syntax) => {
                if syntax == ast::TraitObjectSyntax::Dyn {
                    self.word_space("dyn");
                }
                let mut first = true;
                for bound in bounds {
                    if first {
                        first = false;
                    } else {
                        self.nbsp();
                        self.word_space("+");
                    }
                    self.print_poly_trait_ref(bound);
                }
                if !lifetime.is_elided() {
                    self.nbsp();
                    self.word_space("+");
                    self.print_lifetime(lifetime);
                }
            }
            hir::TyKind::Array(ty, ref length) => {
                self.word("[");
                self.print_type(ty);
                self.word("; ");
                self.print_array_length(length);
                self.word("]");
            }
            hir::TyKind::Typeof(ref e) => {
                self.word("typeof(");
                self.print_anon_const(e);
                self.word(")");
            }
            hir::TyKind::Err => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
            hir::TyKind::Infer => {
                self.word("_");
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self, item: &hir::ForeignItem<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(self.attrs(item.hir_id()));
        match item.kind {
            hir::ForeignItemKind::Fn(decl, arg_names, generics) => {
                self.head("");
                self.print_fn(
                    decl,
                    hir::FnHeader {
                        unsafety: hir::Unsafety::Normal,
                        constness: hir::Constness::NotConst,
                        abi: Abi::Rust,
                        asyncness: hir::IsAsync::NotAsync,
                    },
                    Some(item.ident.name),
                    generics,
                    arg_names,
                    None,
                );
                self.end(); // end head-ibox
                self.word(";");
                self.end() // end the outer fn box
            }
            hir::ForeignItemKind::Static(t, m) => {
                self.head("static");
                if m.is_mut() {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(t);
                self.word(";");
                self.end(); // end the head-ibox
                self.end() // end the outer cbox
            }
            hir::ForeignItemKind::Type => {
                self.head("type");
                self.print_ident(item.ident);
                self.word(";");
                self.end(); // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_const(
        &mut self,
        ident: Ident,
        ty: &hir::Ty<'_>,
        default: Option<hir::BodyId>,
    ) {
        self.head("");
        self.word_space("const");
        self.print_ident(ident);
        self.word_space(":");
        self.print_type(ty);
        if let Some(expr) = default {
            self.space();
            self.word_space("=");
            self.ann.nested(self, Nested::Body(expr));
        }
        self.word(";")
    }

    fn print_associated_type(
        &mut self,
        ident: Ident,
        generics: &hir::Generics<'_>,
        bounds: Option<hir::GenericBounds<'_>>,
        ty: Option<&hir::Ty<'_>>,
    ) {
        self.word_space("type");
        self.print_ident(ident);
        self.print_generic_params(generics.params);
        if let Some(bounds) = bounds {
            self.print_bounds(":", bounds);
        }
        self.print_where_clause(generics);
        if let Some(ty) = ty {
            self.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.word(";")
    }

    fn print_item_type(
        &mut self,
        item: &hir::Item<'_>,
        generics: &hir::Generics<'_>,
        inner: impl Fn(&mut Self),
    ) {
        self.head("type");
        self.print_ident(item.ident);
        self.print_generic_params(generics.params);
        self.end(); // end the inner ibox

        self.print_where_clause(generics);
        self.space();
        inner(self);
        self.word(";");
        self.end(); // end the outer ibox
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &hir::Item<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        let attrs = self.attrs(item.hir_id());
        self.print_outer_attributes(attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.kind {
            hir::ItemKind::ExternCrate(orig_name) => {
                self.head("extern crate");
                if let Some(orig_name) = orig_name {
                    self.print_name(orig_name);
                    self.space();
                    self.word("as");
                    self.space();
                }
                self.print_ident(item.ident);
                self.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            hir::ItemKind::Use(path, kind) => {
                self.head("use");
                self.print_path(path, false);

                match kind {
                    hir::UseKind::Single => {
                        if path.segments.last().unwrap().ident != item.ident {
                            self.space();
                            self.word_space("as");
                            self.print_ident(item.ident);
                        }
                        self.word(";");
                    }
                    hir::UseKind::Glob => self.word("::*;"),
                    hir::UseKind::ListStem => self.word("::{};"),
                }
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            hir::ItemKind::Static(ty, m, expr) => {
                self.head("static");
                if m.is_mut() {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(ty);
                self.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.word(";");
                self.end(); // end the outer cbox
            }
            hir::ItemKind::Const(ty, expr) => {
                self.head("const");
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(ty);
                self.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.word(";");
                self.end(); // end the outer cbox
            }
            hir::ItemKind::Fn(ref sig, param_names, body) => {
                self.head("");
                self.print_fn(
                    sig.decl,
                    sig.header,
                    Some(item.ident.name),
                    param_names,
                    &[],
                    Some(body),
                );
                self.word(" ");
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ItemKind::Macro(ref macro_def, _) => {
                self.print_mac_def(macro_def, &item.ident, item.span, |_| {});
            }
            hir::ItemKind::Mod(ref _mod) => {
                self.head("mod");
                self.print_ident(item.ident);
                self.nbsp();
                self.bopen();
                self.print_mod(_mod, attrs);
                self.bclose(item.span);
            }
            hir::ItemKind::ForeignMod { abi, items } => {
                self.head("extern");
                self.word_nbsp(abi.to_string());
                self.bopen();
                self.print_inner_attributes(self.attrs(item.hir_id()));
                for item in items {
                    self.ann.nested(self, Nested::ForeignItem(item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::GlobalAsm(asm) => {
                self.head("global_asm!");
                self.print_inline_asm(asm);
                self.end()
            }
            hir::ItemKind::TyAlias(ty, generics) => {
                self.print_item_type(item, generics, |state| {
                    state.word_space("=");
                    state.print_type(ty);
                });
            }
            hir::ItemKind::OpaqueTy(ref opaque_ty) => {
                self.print_item_type(item, opaque_ty.generics, |state| {
                    let mut real_bounds = Vec::with_capacity(opaque_ty.bounds.len());
                    for b in opaque_ty.bounds {
                        if let GenericBound::Trait(ptr, hir::TraitBoundModifier::Maybe) = b {
                            state.space();
                            state.word_space("for ?");
                            state.print_trait_ref(&ptr.trait_ref);
                        } else {
                            real_bounds.push(b);
                        }
                    }
                    state.print_bounds("= impl", real_bounds);
                });
            }
            hir::ItemKind::Enum(ref enum_definition, params) => {
                self.print_enum_def(enum_definition, params, item.ident.name, item.span);
            }
            hir::ItemKind::Struct(ref struct_def, generics) => {
                self.head("struct");
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Union(ref struct_def, generics) => {
                self.head("union");
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Impl(&hir::Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                defaultness_span: _,
                generics,
                ref of_trait,
                self_ty,
                items,
            }) => {
                self.head("");
                self.print_defaultness(defaultness);
                self.print_unsafety(unsafety);
                self.word_nbsp("impl");

                if !generics.params.is_empty() {
                    self.print_generic_params(generics.params);
                    self.space();
                }

                if constness == hir::Constness::Const {
                    self.word_nbsp("const");
                }

                if let hir::ImplPolarity::Negative(_) = polarity {
                    self.word("!");
                }

                if let Some(t) = of_trait {
                    self.print_trait_ref(t);
                    self.space();
                    self.word_space("for");
                }

                self.print_type(self_ty);
                self.print_where_clause(generics);

                self.space();
                self.bopen();
                self.print_inner_attributes(attrs);
                for impl_item in items {
                    self.ann.nested(self, Nested::ImplItem(impl_item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::Trait(is_auto, unsafety, generics, bounds, trait_items) => {
                self.head("");
                self.print_is_auto(is_auto);
                self.print_unsafety(unsafety);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds {
                    if let GenericBound::Trait(ptr, hir::TraitBoundModifier::Maybe) = b {
                        self.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b);
                    }
                }
                self.print_bounds(":", real_bounds);
                self.print_where_clause(generics);
                self.word(" ");
                self.bopen();
                for trait_item in trait_items {
                    self.ann.nested(self, Nested::TraitItem(trait_item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::TraitAlias(generics, bounds) => {
                self.head("trait");
                self.print_ident(item.ident);
                self.print_generic_params(generics.params);
                self.nbsp();
                self.print_bounds("=", bounds);
                self.print_where_clause(generics);
                self.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    pub fn print_trait_ref(&mut self, t: &hir::TraitRef<'_>) {
        self.print_path(t.path, false);
    }

    fn print_formal_generic_params(&mut self, generic_params: &[hir::GenericParam<'_>]) {
        if !generic_params.is_empty() {
            self.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef<'_>) {
        self.print_formal_generic_params(t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref);
    }

    pub fn print_enum_def(
        &mut self,
        enum_definition: &hir::EnumDef<'_>,
        generics: &hir::Generics<'_>,
        name: Symbol,
        span: rustc_span::Span,
    ) {
        self.head("enum");
        self.print_name(name);
        self.print_generic_params(generics.params);
        self.print_where_clause(generics);
        self.space();
        self.print_variants(enum_definition.variants, span);
    }

    pub fn print_variants(&mut self, variants: &[hir::Variant<'_>], span: rustc_span::Span) {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(self.attrs(v.hir_id));
            self.ibox(INDENT_UNIT);
            self.print_variant(v);
            self.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span)
    }

    pub fn print_defaultness(&mut self, defaultness: hir::Defaultness) {
        match defaultness {
            hir::Defaultness::Default { .. } => self.word_nbsp("default"),
            hir::Defaultness::Final => (),
        }
    }

    pub fn print_struct(
        &mut self,
        struct_def: &hir::VariantData<'_>,
        generics: &hir::Generics<'_>,
        name: Symbol,
        span: rustc_span::Span,
        print_finalizer: bool,
    ) {
        self.print_name(name);
        self.print_generic_params(generics.params);
        match struct_def {
            hir::VariantData::Tuple(..) | hir::VariantData::Unit(..) => {
                if let hir::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                        s.maybe_print_comment(field.span.lo());
                        s.print_outer_attributes(s.attrs(field.hir_id));
                        s.print_type(field.ty);
                    });
                    self.pclose();
                }
                self.print_where_clause(generics);
                if print_finalizer {
                    self.word(";");
                }
                self.end();
                self.end() // close the outer-box
            }
            hir::VariantData::Struct(..) => {
                self.print_where_clause(generics);
                self.nbsp();
                self.bopen();
                self.hardbreak_if_not_bol();

                for field in struct_def.fields() {
                    self.hardbreak_if_not_bol();
                    self.maybe_print_comment(field.span.lo());
                    self.print_outer_attributes(self.attrs(field.hir_id));
                    self.print_ident(field.ident);
                    self.word_nbsp(":");
                    self.print_type(field.ty);
                    self.word(",");
                }

                self.bclose(span)
            }
        }
    }

    pub fn print_variant(&mut self, v: &hir::Variant<'_>) {
        self.head("");
        let generics = hir::Generics::empty();
        self.print_struct(&v.data, generics, v.ident.name, v.span, false);
        if let Some(ref d) = v.disr_expr {
            self.space();
            self.word_space("=");
            self.print_anon_const(d);
        }
    }
    pub fn print_method_sig(
        &mut self,
        ident: Ident,
        m: &hir::FnSig<'_>,
        generics: &hir::Generics<'_>,
        arg_names: &[Ident],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn(m.decl, m.header, Some(ident.name), generics, arg_names, body_id);
    }

    pub fn print_trait_item(&mut self, ti: &hir::TraitItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ti.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ti.span.lo());
        self.print_outer_attributes(self.attrs(ti.hir_id()));
        match ti.kind {
            hir::TraitItemKind::Const(ty, default) => {
                self.print_associated_const(ti.ident, ty, default);
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(arg_names)) => {
                self.print_method_sig(ti.ident, sig, ti.generics, arg_names, None);
                self.word(";");
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                self.head("");
                self.print_method_sig(ti.ident, sig, ti.generics, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::TraitItemKind::Type(bounds, default) => {
                self.print_associated_type(ti.ident, ti.generics, Some(bounds), default);
            }
        }
        self.ann.post(self, AnnNode::SubItem(ti.hir_id()))
    }

    pub fn print_impl_item(&mut self, ii: &hir::ImplItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ii.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ii.span.lo());
        self.print_outer_attributes(self.attrs(ii.hir_id()));

        match ii.kind {
            hir::ImplItemKind::Const(ty, expr) => {
                self.print_associated_const(ii.ident, ty, Some(expr));
            }
            hir::ImplItemKind::Fn(ref sig, body) => {
                self.head("");
                self.print_method_sig(ii.ident, sig, ii.generics, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ImplItemKind::Type(ty) => {
                self.print_associated_type(ii.ident, ii.generics, None, Some(ty));
            }
        }
        self.ann.post(self, AnnNode::SubItem(ii.hir_id()))
    }

    pub fn print_local(
        &mut self,
        init: Option<&hir::Expr<'_>>,
        els: Option<&hir::Block<'_>>,
        decl: impl Fn(&mut Self),
    ) {
        self.space_if_not_bol();
        self.ibox(INDENT_UNIT);
        self.word_nbsp("let");

        self.ibox(INDENT_UNIT);
        decl(self);
        self.end();

        if let Some(init) = init {
            self.nbsp();
            self.word_space("=");
            self.print_expr(init);
        }

        if let Some(els) = els {
            self.nbsp();
            self.word_space("else");
            // containing cbox, will be closed by print-block at `}`
            self.cbox(0);
            // head-box, will be closed by print-block after `{`
            self.ibox(0);
            self.print_block(els);
        }

        self.end()
    }

    pub fn print_stmt(&mut self, st: &hir::Stmt<'_>) {
        self.maybe_print_comment(st.span.lo());
        match st.kind {
            hir::StmtKind::Local(loc) => {
                self.print_local(loc.init, loc.els, |this| this.print_local_decl(loc));
            }
            hir::StmtKind::Item(item) => self.ann.nested(self, Nested::Item(item)),
            hir::StmtKind::Expr(expr) => {
                self.space_if_not_bol();
                self.print_expr(expr);
            }
            hir::StmtKind::Semi(expr) => {
                self.space_if_not_bol();
                self.print_expr(expr);
                self.word(";");
            }
        }
        if stmt_ends_with_semi(&st.kind) {
            self.word(";");
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &hir::Block<'_>) {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &hir::Block<'_>) {
        self.print_block_maybe_unclosed(blk, &[], false)
    }

    pub fn print_block_with_attrs(&mut self, blk: &hir::Block<'_>, attrs: &[ast::Attribute]) {
        self.print_block_maybe_unclosed(blk, attrs, true)
    }

    pub fn print_block_maybe_unclosed(
        &mut self,
        blk: &hir::Block<'_>,
        attrs: &[ast::Attribute],
        close_box: bool,
    ) {
        match blk.rules {
            hir::BlockCheckMode::UnsafeBlock(..) => self.word_space("unsafe"),
            hir::BlockCheckMode::DefaultBlock => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen();

        self.print_inner_attributes(attrs);

        for st in blk.stmts {
            self.print_stmt(st);
        }
        if let Some(expr) = blk.expr {
            self.space_if_not_bol();
            self.print_expr(expr);
            self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
        }
        self.bclose_maybe_open(blk.span, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    fn print_else(&mut self, els: Option<&hir::Expr<'_>>) {
        if let Some(els_inner) = els {
            match els_inner.kind {
                // Another `else if` block.
                hir::ExprKind::If(i, then, e) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.word(" else if ");
                    self.print_expr_as_cond(i);
                    self.space();
                    self.print_expr(then);
                    self.print_else(e);
                }
                // Final `else` block.
                hir::ExprKind::Block(b, _) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.word(" else ");
                    self.print_block(b);
                }
                // Constraints would be great here!
                _ => {
                    panic!("print_if saw if with weird alternative");
                }
            }
        }
    }

    pub fn print_if(
        &mut self,
        test: &hir::Expr<'_>,
        blk: &hir::Expr<'_>,
        elseopt: Option<&hir::Expr<'_>>,
    ) {
        self.head("if");
        self.print_expr_as_cond(test);
        self.space();
        self.print_expr(blk);
        self.print_else(elseopt)
    }

    pub fn print_array_length(&mut self, len: &hir::ArrayLen) {
        match len {
            hir::ArrayLen::Infer(_, _) => self.word("_"),
            hir::ArrayLen::Body(ct) => self.print_anon_const(ct),
        }
    }

    pub fn print_anon_const(&mut self, constant: &hir::AnonConst) {
        self.ann.nested(self, Nested::Body(constant.body))
    }

    fn print_call_post(&mut self, args: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    fn print_expr_maybe_paren(&mut self, expr: &hir::Expr<'_>, prec: i8) {
        self.print_expr_cond_paren(expr, expr.precedence().order() < prec)
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    pub fn print_expr_as_cond(&mut self, expr: &hir::Expr<'_>) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    fn print_expr_cond_paren(&mut self, expr: &hir::Expr<'_>, needs_par: bool) {
        if needs_par {
            self.popen();
        }
        if let hir::ExprKind::DropTemps(actual_expr) = expr.kind {
            self.print_expr(actual_expr);
        } else {
            self.print_expr(expr);
        }
        if needs_par {
            self.pclose();
        }
    }

    /// Print a `let pat = expr` expression.
    fn print_let(&mut self, pat: &hir::Pat<'_>, ty: Option<&hir::Ty<'_>>, init: &hir::Expr<'_>) {
        self.word_space("let");
        self.print_pat(pat);
        if let Some(ty) = ty {
            self.word_space(":");
            self.print_type(ty);
        }
        self.space();
        self.word_space("=");
        let npals = || parser::needs_par_as_let_scrutinee(init.precedence().order());
        self.print_expr_cond_paren(init, Self::cond_needs_par(init) || npals())
    }

    // Does `expr` need parentheses when printed in a condition position?
    //
    // These cases need parens due to the parse error observed in #26461: `if return {}`
    // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
    fn cond_needs_par(expr: &hir::Expr<'_>) -> bool {
        match expr.kind {
            hir::ExprKind::Break(..) | hir::ExprKind::Closure { .. } | hir::ExprKind::Ret(..) => {
                true
            }
            _ => contains_exterior_struct_lit(expr),
        }
    }

    fn print_expr_vec(&mut self, exprs: &[hir::Expr<'_>]) {
        self.ibox(INDENT_UNIT);
        self.word("[");
        self.commasep_exprs(Inconsistent, exprs);
        self.word("]");
        self.end()
    }

    fn print_expr_anon_const(&mut self, anon_const: &hir::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.word_space("const");
        self.print_anon_const(anon_const);
        self.end()
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr<'_>, count: &hir::ArrayLen) {
        self.ibox(INDENT_UNIT);
        self.word("[");
        self.print_expr(element);
        self.word_space(";");
        self.print_array_length(count);
        self.word("]");
        self.end()
    }

    fn print_expr_struct(
        &mut self,
        qpath: &hir::QPath<'_>,
        fields: &[hir::ExprField<'_>],
        wth: Option<&hir::Expr<'_>>,
    ) {
        self.print_qpath(qpath, true);
        self.word("{");
        self.commasep_cmnt(Consistent, fields, |s, field| s.print_expr_field(field), |f| f.span);
        if let Some(expr) = wth {
            self.ibox(INDENT_UNIT);
            if !fields.is_empty() {
                self.word(",");
                self.space();
            }
            self.word("..");
            self.print_expr(expr);
            self.end();
        } else if !fields.is_empty() {
            self.word(",");
        }

        self.word("}");
    }

    fn print_expr_field(&mut self, field: &hir::ExprField<'_>) {
        if self.attrs(field.hir_id).is_empty() {
            self.space();
        }
        self.cbox(INDENT_UNIT);
        self.print_outer_attributes(&self.attrs(field.hir_id));
        if !field.is_shorthand {
            self.print_ident(field.ident);
            self.word_space(":");
        }
        self.print_expr(&field.expr);
        self.end()
    }

    fn print_expr_tup(&mut self, exprs: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, exprs);
        if exprs.len() == 1 {
            self.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
        let prec = match func.kind {
            hir::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
            _ => parser::PREC_POSTFIX,
        };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(
        &mut self,
        segment: &hir::PathSegment<'_>,
        receiver: &hir::Expr<'_>,
        args: &[hir::Expr<'_>],
    ) {
        let base_args = args;
        self.print_expr_maybe_paren(&receiver, parser::PREC_POSTFIX);
        self.word(".");
        self.print_ident(segment.ident);

        let generic_args = segment.args();
        if !generic_args.args.is_empty() || !generic_args.bindings.is_empty() {
            self.print_generic_args(generic_args, true);
        }

        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self, op: hir::BinOp, lhs: &hir::Expr<'_>, rhs: &hir::Expr<'_>) {
        let assoc_op = bin_op_to_assoc_op(op.node);
        let prec = assoc_op.precedence() as i8;
        let fixity = assoc_op.fixity();

        let (left_prec, right_prec) = match fixity {
            Fixity::Left => (prec, prec + 1),
            Fixity::Right => (prec + 1, prec),
            Fixity::None => (prec + 1, prec + 1),
        };

        let left_prec = match (&lhs.kind, op.node) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&hir::ExprKind::Cast { .. }, hir::BinOpKind::Lt | hir::BinOpKind::Shl) => {
                parser::PREC_FORCE_PAREN
            }
            (&hir::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(prec) => {
                parser::PREC_FORCE_PAREN
            }
            _ => left_prec,
        };

        self.print_expr_maybe_paren(lhs, left_prec);
        self.space();
        self.word_space(op.node.as_str());
        self.print_expr_maybe_paren(rhs, right_prec)
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr<'_>) {
        self.word(op.as_str());
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(
        &mut self,
        kind: hir::BorrowKind,
        mutability: hir::Mutability,
        expr: &hir::Expr<'_>,
    ) {
        self.word("&");
        match kind {
            hir::BorrowKind::Ref => self.print_mutability(mutability, false),
            hir::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_literal(&mut self, lit: &hir::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.word(lit.node.to_string())
    }

    fn print_inline_asm(&mut self, asm: &hir::InlineAsm<'_>) {
        enum AsmArg<'a> {
            Template(String),
            Operand(&'a hir::InlineAsmOperand<'a>),
            Options(ast::InlineAsmOptions),
        }

        let mut args = vec![AsmArg::Template(ast::InlineAsmTemplatePiece::to_string(asm.template))];
        args.extend(asm.operands.iter().map(|(o, _)| AsmArg::Operand(o)));
        if !asm.options.is_empty() {
            args.push(AsmArg::Options(asm.options));
        }

        self.popen();
        self.commasep(Consistent, &args, |s, arg| match *arg {
            AsmArg::Template(ref template) => s.print_string(template, ast::StrStyle::Cooked),
            AsmArg::Operand(op) => match *op {
                hir::InlineAsmOperand::In { reg, ref expr } => {
                    s.word("in");
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::Out { reg, late, ref expr } => {
                    s.word(if late { "lateout" } else { "out" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    match expr {
                        Some(expr) => s.print_expr(expr),
                        None => s.word("_"),
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late, ref expr } => {
                    s.word(if late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::SplitInOut { reg, late, ref in_expr, ref out_expr } => {
                    s.word(if late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(in_expr);
                    s.space();
                    s.word_space("=>");
                    match out_expr {
                        Some(out_expr) => s.print_expr(out_expr),
                        None => s.word("_"),
                    }
                }
                hir::InlineAsmOperand::Const { ref anon_const } => {
                    s.word("const");
                    s.space();
                    s.print_anon_const(anon_const);
                }
                hir::InlineAsmOperand::SymFn { ref anon_const } => {
                    s.word("sym_fn");
                    s.space();
                    s.print_anon_const(anon_const);
                }
                hir::InlineAsmOperand::SymStatic { ref path, def_id: _ } => {
                    s.word("sym_static");
                    s.space();
                    s.print_qpath(path, true);
                }
            },
            AsmArg::Options(opts) => {
                s.word("options");
                s.popen();
                let mut options = vec![];
                if opts.contains(ast::InlineAsmOptions::PURE) {
                    options.push("pure");
                }
                if opts.contains(ast::InlineAsmOptions::NOMEM) {
                    options.push("nomem");
                }
                if opts.contains(ast::InlineAsmOptions::READONLY) {
                    options.push("readonly");
                }
                if opts.contains(ast::InlineAsmOptions::PRESERVES_FLAGS) {
                    options.push("preserves_flags");
                }
                if opts.contains(ast::InlineAsmOptions::NORETURN) {
                    options.push("noreturn");
                }
                if opts.contains(ast::InlineAsmOptions::NOSTACK) {
                    options.push("nostack");
                }
                if opts.contains(ast::InlineAsmOptions::ATT_SYNTAX) {
                    options.push("att_syntax");
                }
                if opts.contains(ast::InlineAsmOptions::RAW) {
                    options.push("raw");
                }
                if opts.contains(ast::InlineAsmOptions::MAY_UNWIND) {
                    options.push("may_unwind");
                }
                s.commasep(Inconsistent, &options, |s, &opt| {
                    s.word(opt);
                });
                s.pclose();
            }
        });
        self.pclose();
    }

    pub fn print_expr(&mut self, expr: &hir::Expr<'_>) {
        self.maybe_print_comment(expr.span.lo());
        self.print_outer_attributes(self.attrs(expr.hir_id));
        self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.kind {
            hir::ExprKind::Box(expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            hir::ExprKind::Array(exprs) => {
                self.print_expr_vec(exprs);
            }
            hir::ExprKind::ConstBlock(ref anon_const) => {
                self.print_expr_anon_const(anon_const);
            }
            hir::ExprKind::Repeat(element, ref count) => {
                self.print_expr_repeat(element, count);
            }
            hir::ExprKind::Struct(qpath, fields, wth) => {
                self.print_expr_struct(qpath, fields, wth);
            }
            hir::ExprKind::Tup(exprs) => {
                self.print_expr_tup(exprs);
            }
            hir::ExprKind::Call(func, args) => {
                self.print_expr_call(func, args);
            }
            hir::ExprKind::MethodCall(segment, receiver, args, _) => {
                self.print_expr_method_call(segment, receiver, args);
            }
            hir::ExprKind::Binary(op, lhs, rhs) => {
                self.print_expr_binary(op, lhs, rhs);
            }
            hir::ExprKind::Unary(op, expr) => {
                self.print_expr_unary(op, expr);
            }
            hir::ExprKind::AddrOf(k, m, expr) => {
                self.print_expr_addr_of(k, m, expr);
            }
            hir::ExprKind::Lit(ref lit) => {
                self.print_literal(lit);
            }
            hir::ExprKind::Cast(expr, ty) => {
                let prec = AssocOp::As.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.space();
                self.word_space("as");
                self.print_type(ty);
            }
            hir::ExprKind::Type(expr, ty) => {
                let prec = AssocOp::Colon.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.word_space(":");
                self.print_type(ty);
            }
            hir::ExprKind::DropTemps(init) => {
                // Print `{`:
                self.cbox(INDENT_UNIT);
                self.ibox(0);
                self.bopen();

                // Print `let _t = $init;`:
                let temp = Ident::from_str("_t");
                self.print_local(Some(init), None, |this| this.print_ident(temp));
                self.word(";");

                // Print `_t`:
                self.space_if_not_bol();
                self.print_ident(temp);

                // Print `}`:
                self.bclose_maybe_open(expr.span, true);
            }
            hir::ExprKind::Let(&hir::Let { pat, ty, init, .. }) => {
                self.print_let(pat, ty, init);
            }
            hir::ExprKind::If(test, blk, elseopt) => {
                self.print_if(test, blk, elseopt);
            }
            hir::ExprKind::Loop(blk, opt_label, _, _) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("loop");
                self.print_block(blk);
            }
            hir::ExprKind::Match(expr, arms, _) => {
                self.cbox(INDENT_UNIT);
                self.ibox(INDENT_UNIT);
                self.word_nbsp("match");
                self.print_expr_as_cond(expr);
                self.space();
                self.bopen();
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose(expr.span);
            }
            hir::ExprKind::Closure(&hir::Closure {
                binder,
                constness,
                capture_clause,
                bound_generic_params,
                fn_decl,
                body,
                fn_decl_span: _,
                fn_arg_span: _,
                movability: _,
                def_id: _,
            }) => {
                self.print_closure_binder(binder, bound_generic_params);
                self.print_constness(constness);
                self.print_capture_clause(capture_clause);

                self.print_closure_params(fn_decl, body);
                self.space();

                // This is a bare expression.
                self.ann.nested(self, Nested::Body(body));
                self.end(); // need to close a box

                // A box will be closed by `print_expr`, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            hir::ExprKind::Block(blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at `}`
                self.cbox(INDENT_UNIT);
                // head-box, will be closed by print-block after `{`
                self.ibox(0);
                self.print_block(blk);
            }
            hir::ExprKind::Assign(lhs, rhs, _) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.space();
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            hir::ExprKind::AssignOp(op, lhs, rhs) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.space();
                self.word(op.node.as_str());
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            hir::ExprKind::Field(expr, ident) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.word(".");
                self.print_ident(ident);
            }
            hir::ExprKind::Index(expr, index) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.word("[");
                self.print_expr(index);
                self.word("]");
            }
            hir::ExprKind::Path(ref qpath) => self.print_qpath(qpath, true),
            hir::ExprKind::Break(destination, opt_expr) => {
                self.word("break");
                if let Some(label) = destination.label {
                    self.space();
                    self.print_ident(label.ident);
                }
                if let Some(expr) = opt_expr {
                    self.space();
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            hir::ExprKind::Continue(destination) => {
                self.word("continue");
                if let Some(label) = destination.label {
                    self.space();
                    self.print_ident(label.ident);
                }
            }
            hir::ExprKind::Ret(result) => {
                self.word("return");
                if let Some(expr) = result {
                    self.word(" ");
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            hir::ExprKind::InlineAsm(asm) => {
                self.word("asm!");
                self.print_inline_asm(asm);
            }
            hir::ExprKind::Yield(expr, _) => {
                self.word_space("yield");
                self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
            }
            hir::ExprKind::Err => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
        }
        self.ann.post(self, AnnNode::Expr(expr));
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &hir::Local<'_>) {
        self.print_pat(loc.pat);
        if let Some(ty) = loc.ty {
            self.word_space(":");
            self.print_type(ty);
        }
    }

    pub fn print_name(&mut self, name: Symbol) {
        self.print_ident(Ident::with_dummy_span(name))
    }

    pub fn print_path<R>(&mut self, path: &hir::Path<'_, R>, colons_before_params: bool) {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                self.word("::")
            }
            if segment.ident.name != kw::PathRoot {
                self.print_ident(segment.ident);
                self.print_generic_args(segment.args(), colons_before_params);
            }
        }
    }

    pub fn print_path_segment(&mut self, segment: &hir::PathSegment<'_>) {
        if segment.ident.name != kw::PathRoot {
            self.print_ident(segment.ident);
            self.print_generic_args(segment.args(), false);
        }
    }

    pub fn print_qpath(&mut self, qpath: &hir::QPath<'_>, colons_before_params: bool) {
        match *qpath {
            hir::QPath::Resolved(None, path) => self.print_path(path, colons_before_params),
            hir::QPath::Resolved(Some(qself), path) => {
                self.word("<");
                self.print_type(qself);
                self.space();
                self.word_space("as");

                for (i, segment) in path.segments[..path.segments.len() - 1].iter().enumerate() {
                    if i > 0 {
                        self.word("::")
                    }
                    if segment.ident.name != kw::PathRoot {
                        self.print_ident(segment.ident);
                        self.print_generic_args(segment.args(), colons_before_params);
                    }
                }

                self.word(">");
                self.word("::");
                let item_segment = path.segments.last().unwrap();
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.args(), colons_before_params)
            }
            hir::QPath::TypeRelative(qself, item_segment) => {
                // If we've got a compound-qualified-path, let's push an additional pair of angle
                // brackets, so that we pretty-print `<<A::B>::C>` as `<A::B>::C`, instead of just
                // `A::B::C` (since the latter could be ambiguous to the user)
                if let hir::TyKind::Path(hir::QPath::Resolved(None, _)) = qself.kind {
                    self.print_type(qself);
                } else {
                    self.word("<");
                    self.print_type(qself);
                    self.word(">");
                }

                self.word("::");
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.args(), colons_before_params)
            }
            hir::QPath::LangItem(lang_item, span, _) => {
                self.word("#[lang = \"");
                self.print_ident(Ident::new(lang_item.name(), span));
                self.word("\"]");
            }
        }
    }

    fn print_generic_args(
        &mut self,
        generic_args: &hir::GenericArgs<'_>,
        colons_before_params: bool,
    ) {
        if generic_args.parenthesized {
            self.word("(");
            self.commasep(Inconsistent, generic_args.inputs(), |s, ty| s.print_type(ty));
            self.word(")");

            self.space_if_not_bol();
            self.word_space("->");
            self.print_type(generic_args.bindings[0].ty());
        } else {
            let start = if colons_before_params { "::<" } else { "<" };
            let empty = Cell::new(true);
            let start_or_comma = |this: &mut Self| {
                if empty.get() {
                    empty.set(false);
                    this.word(start)
                } else {
                    this.word_space(",")
                }
            };

            let mut nonelided_generic_args: bool = false;
            let elide_lifetimes = generic_args.args.iter().all(|arg| match arg {
                GenericArg::Lifetime(lt) if lt.is_elided() => true,
                GenericArg::Lifetime(_) => {
                    nonelided_generic_args = true;
                    false
                }
                _ => {
                    nonelided_generic_args = true;
                    true
                }
            });

            if nonelided_generic_args {
                start_or_comma(self);
                self.commasep(
                    Inconsistent,
                    generic_args.args,
                    |s, generic_arg| match generic_arg {
                        GenericArg::Lifetime(lt) if !elide_lifetimes => s.print_lifetime(lt),
                        GenericArg::Lifetime(_) => {}
                        GenericArg::Type(ty) => s.print_type(ty),
                        GenericArg::Const(ct) => s.print_anon_const(&ct.value),
                        GenericArg::Infer(_inf) => s.word("_"),
                    },
                );
            }

            for binding in generic_args.bindings {
                start_or_comma(self);
                self.print_type_binding(binding);
            }

            if !empty.get() {
                self.word(">")
            }
        }
    }

    pub fn print_type_binding(&mut self, binding: &hir::TypeBinding<'_>) {
        self.print_ident(binding.ident);
        self.print_generic_args(binding.gen_args, false);
        self.space();
        match binding.kind {
            hir::TypeBindingKind::Equality { ref term } => {
                self.word_space("=");
                match term {
                    Term::Ty(ty) => self.print_type(ty),
                    Term::Const(ref c) => self.print_anon_const(c),
                }
            }
            hir::TypeBindingKind::Constraint { bounds } => {
                self.print_bounds(":", bounds);
            }
        }
    }

    pub fn print_pat(&mut self, pat: &hir::Pat<'_>) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        // Pat isn't normalized, but the beauty of it
        // is that it doesn't matter
        match pat.kind {
            PatKind::Wild => self.word("_"),
            PatKind::Binding(BindingAnnotation(by_ref, mutbl), _, ident, sub) => {
                if by_ref == ByRef::Yes {
                    self.word_nbsp("ref");
                }
                if mutbl.is_mut() {
                    self.word_nbsp("mut");
                }
                self.print_ident(ident);
                if let Some(p) = sub {
                    self.word("@");
                    self.print_pat(p);
                }
            }
            PatKind::TupleStruct(ref qpath, elts, ddpos) => {
                self.print_qpath(qpath, true);
                self.popen();
                if let Some(ddpos) = ddpos.as_opt_usize() {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.word("..");
                    if ddpos != elts.len() {
                        self.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                }
                self.pclose();
            }
            PatKind::Path(ref qpath) => {
                self.print_qpath(qpath, true);
            }
            PatKind::Struct(ref qpath, fields, etc) => {
                self.print_qpath(qpath, true);
                self.nbsp();
                self.word("{");
                let empty = fields.is_empty() && !etc;
                if !empty {
                    self.space();
                }
                self.commasep_cmnt(Consistent, &fields, |s, f| s.print_patfield(f), |f| f.pat.span);
                if etc {
                    if !fields.is_empty() {
                        self.word_space(",");
                    }
                    self.word("..");
                }
                if !empty {
                    self.space();
                }
                self.word("}");
            }
            PatKind::Or(pats) => {
                self.strsep("|", true, Inconsistent, pats, |s, p| s.print_pat(p));
            }
            PatKind::Tuple(elts, ddpos) => {
                self.popen();
                if let Some(ddpos) = ddpos.as_opt_usize() {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.word("..");
                    if ddpos != elts.len() {
                        self.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                    if elts.len() == 1 {
                        self.word(",");
                    }
                }
                self.pclose();
            }
            PatKind::Box(inner) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.word("box ");
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Ref(inner, mutbl) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.word("&");
                self.word(mutbl.prefix_str());
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Lit(e) => self.print_expr(e),
            PatKind::Range(begin, end, end_kind) => {
                if let Some(expr) = begin {
                    self.print_expr(expr);
                }
                match end_kind {
                    RangeEnd::Included => self.word("..."),
                    RangeEnd::Excluded => self.word(".."),
                }
                if let Some(expr) = end {
                    self.print_expr(expr);
                }
            }
            PatKind::Slice(before, slice, after) => {
                self.word("[");
                self.commasep(Inconsistent, before, |s, p| s.print_pat(p));
                if let Some(p) = slice {
                    if !before.is_empty() {
                        self.word_space(",");
                    }
                    if let PatKind::Wild = p.kind {
                        // Print nothing.
                    } else {
                        self.print_pat(p);
                    }
                    self.word("..");
                    if !after.is_empty() {
                        self.word_space(",");
                    }
                }
                self.commasep(Inconsistent, after, |s, p| s.print_pat(p));
                self.word("]");
            }
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    pub fn print_patfield(&mut self, field: &hir::PatField<'_>) {
        if self.attrs(field.hir_id).is_empty() {
            self.space();
        }
        self.cbox(INDENT_UNIT);
        self.print_outer_attributes(&self.attrs(field.hir_id));
        if !field.is_shorthand {
            self.print_ident(field.ident);
            self.word_nbsp(":");
        }
        self.print_pat(field.pat);
        self.end();
    }

    pub fn print_param(&mut self, arg: &hir::Param<'_>) {
        self.print_outer_attributes(self.attrs(arg.hir_id));
        self.print_pat(arg.pat);
    }

    pub fn print_arm(&mut self, arm: &hir::Arm<'_>) {
        // I have no idea why this check is necessary, but here it
        // is :(
        if self.attrs(arm.hir_id).is_empty() {
            self.space();
        }
        self.cbox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Arm(arm));
        self.ibox(0);
        self.print_outer_attributes(self.attrs(arm.hir_id));
        self.print_pat(arm.pat);
        self.space();
        if let Some(ref g) = arm.guard {
            match *g {
                hir::Guard::If(e) => {
                    self.word_space("if");
                    self.print_expr(e);
                    self.space();
                }
                hir::Guard::IfLet(&hir::Let { pat, ty, init, .. }) => {
                    self.word_nbsp("if");
                    self.print_let(pat, ty, init);
                }
            }
        }
        self.word_space("=>");

        match arm.body.kind {
            hir::ExprKind::Block(blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // the block will close the pattern's ibox
                self.print_block_unclosed(blk);

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = blk.rules
                {
                    self.word(",");
                }
            }
            _ => {
                self.end(); // close the ibox for the pattern
                self.print_expr(arm.body);
                self.word(",");
            }
        }
        self.ann.post(self, AnnNode::Arm(arm));
        self.end() // close enclosing cbox
    }

    pub fn print_fn(
        &mut self,
        decl: &hir::FnDecl<'_>,
        header: hir::FnHeader,
        name: Option<Symbol>,
        generics: &hir::Generics<'_>,
        arg_names: &[Ident],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn_header_info(header);

        if let Some(name) = name {
            self.nbsp();
            self.print_name(name);
        }
        self.print_generic_params(generics.params);

        self.popen();
        let mut i = 0;
        // Make sure we aren't supplied *both* `arg_names` and `body_id`.
        assert!(arg_names.is_empty() || body_id.is_none());
        self.commasep(Inconsistent, decl.inputs, |s, ty| {
            s.ibox(INDENT_UNIT);
            if let Some(arg_name) = arg_names.get(i) {
                s.word(arg_name.to_string());
                s.word(":");
                s.space();
            } else if let Some(body_id) = body_id {
                s.ann.nested(s, Nested::BodyParamPat(body_id, i));
                s.word(":");
                s.space();
            }
            i += 1;
            s.print_type(ty);
            s.end()
        });
        if decl.c_variadic {
            self.word(", ...");
        }
        self.pclose();

        self.print_fn_output(decl);
        self.print_where_clause(generics)
    }

    fn print_closure_params(&mut self, decl: &hir::FnDecl<'_>, body_id: hir::BodyId) {
        self.word("|");
        let mut i = 0;
        self.commasep(Inconsistent, decl.inputs, |s, ty| {
            s.ibox(INDENT_UNIT);

            s.ann.nested(s, Nested::BodyParamPat(body_id, i));
            i += 1;

            if let hir::TyKind::Infer = ty.kind {
                // Print nothing.
            } else {
                s.word(":");
                s.space();
                s.print_type(ty);
            }
            s.end();
        });
        self.word("|");

        if let hir::FnRetTy::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.word_space("->");
        match decl.output {
            hir::FnRetTy::Return(ty) => {
                self.print_type(ty);
                self.maybe_print_comment(ty.span.lo());
            }
            hir::FnRetTy::DefaultReturn(..) => unreachable!(),
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: hir::CaptureBy) {
        match capture_clause {
            hir::CaptureBy::Value => self.word_space("move"),
            hir::CaptureBy::Ref => {}
        }
    }

    pub fn print_closure_binder(
        &mut self,
        binder: hir::ClosureBinder,
        generic_params: &[GenericParam<'_>],
    ) {
        let generic_params = generic_params
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    GenericParam {
                        kind: GenericParamKind::Lifetime { kind: LifetimeParamKind::Explicit },
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();

        match binder {
            hir::ClosureBinder::Default => {}
            // we need to distinguish `|...| {}` from `for<> |...| {}` as `for<>` adds additional restrictions
            hir::ClosureBinder::For { .. } if generic_params.is_empty() => self.word("for<>"),
            hir::ClosureBinder::For { .. } => {
                self.word("for");
                self.word("<");

                self.commasep(Inconsistent, &generic_params, |s, param| {
                    s.print_generic_param(param)
                });

                self.word(">");
                self.nbsp();
            }
        }
    }

    pub fn print_bounds<'b>(
        &mut self,
        prefix: &'static str,
        bounds: impl IntoIterator<Item = &'b hir::GenericBound<'b>>,
    ) {
        let mut first = true;
        for bound in bounds {
            if first {
                self.word(prefix);
            }
            if !(first && prefix.is_empty()) {
                self.nbsp();
            }
            if first {
                first = false;
            } else {
                self.word_space("+");
            }

            match bound {
                GenericBound::Trait(tref, modifier) => {
                    if modifier == &TraitBoundModifier::Maybe {
                        self.word("?");
                    }
                    self.print_poly_trait_ref(tref);
                }
                GenericBound::LangItemTrait(lang_item, span, ..) => {
                    self.word("#[lang = \"");
                    self.print_ident(Ident::new(lang_item.name(), *span));
                    self.word("\"]");
                }
                GenericBound::Outlives(lt) => {
                    self.print_lifetime(lt);
                }
            }
        }
    }

    pub fn print_generic_params(&mut self, generic_params: &[GenericParam<'_>]) {
        if !generic_params.is_empty() {
            self.word("<");

            self.commasep(Inconsistent, generic_params, |s, param| s.print_generic_param(param));

            self.word(">");
        }
    }

    pub fn print_generic_param(&mut self, param: &GenericParam<'_>) {
        if let GenericParamKind::Const { .. } = param.kind {
            self.word_space("const");
        }

        self.print_ident(param.name.ident());

        match param.kind {
            GenericParamKind::Lifetime { .. } => {}
            GenericParamKind::Type { default, .. } => {
                if let Some(default) = default {
                    self.space();
                    self.word_space("=");
                    self.print_type(default);
                }
            }
            GenericParamKind::Const { ty, ref default } => {
                self.word_space(":");
                self.print_type(ty);
                if let Some(default) = default {
                    self.space();
                    self.word_space("=");
                    self.print_anon_const(default);
                }
            }
        }
    }

    pub fn print_lifetime(&mut self, lifetime: &hir::Lifetime) {
        self.print_ident(lifetime.ident)
    }

    pub fn print_where_clause(&mut self, generics: &hir::Generics<'_>) {
        if generics.predicates.is_empty() {
            return;
        }

        self.space();
        self.word_space("where");

        for (i, predicate) in generics.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",");
            }

            match *predicate {
                hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                    bound_generic_params,
                    bounded_ty,
                    bounds,
                    ..
                }) => {
                    self.print_formal_generic_params(bound_generic_params);
                    self.print_type(bounded_ty);
                    self.print_bounds(":", bounds);
                }
                hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                    ref lifetime,
                    bounds,
                    ..
                }) => {
                    self.print_lifetime(lifetime);
                    self.word(":");

                    for (i, bound) in bounds.iter().enumerate() {
                        match bound {
                            GenericBound::Outlives(lt) => {
                                self.print_lifetime(lt);
                            }
                            _ => panic!(),
                        }

                        if i != 0 {
                            self.word(":");
                        }
                    }
                }
                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    lhs_ty, rhs_ty, ..
                }) => {
                    self.print_type(lhs_ty);
                    self.space();
                    self.word_space("=");
                    self.print_type(rhs_ty);
                }
            }
        }
    }

    pub fn print_mutability(&mut self, mutbl: hir::Mutability, print_const: bool) {
        match mutbl {
            hir::Mutability::Mut => self.word_nbsp("mut"),
            hir::Mutability::Not => {
                if print_const {
                    self.word_nbsp("const")
                }
            }
        }
    }

    pub fn print_mt(&mut self, mt: &hir::MutTy<'_>, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(mt.ty);
    }

    pub fn print_fn_output(&mut self, decl: &hir::FnDecl<'_>) {
        if let hir::FnRetTy::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.ibox(INDENT_UNIT);
        self.word_space("->");
        match decl.output {
            hir::FnRetTy::DefaultReturn(..) => unreachable!(),
            hir::FnRetTy::Return(ty) => self.print_type(ty),
        }
        self.end();

        if let hir::FnRetTy::Return(output) = decl.output {
            self.maybe_print_comment(output.span.lo());
        }
    }

    pub fn print_ty_fn(
        &mut self,
        abi: Abi,
        unsafety: hir::Unsafety,
        decl: &hir::FnDecl<'_>,
        name: Option<Symbol>,
        generic_params: &[hir::GenericParam<'_>],
        arg_names: &[Ident],
    ) {
        self.ibox(INDENT_UNIT);
        self.print_formal_generic_params(generic_params);
        let generics = hir::Generics::empty();
        self.print_fn(
            decl,
            hir::FnHeader {
                unsafety,
                abi,
                constness: hir::Constness::NotConst,
                asyncness: hir::IsAsync::NotAsync,
            },
            name,
            generics,
            arg_names,
            None,
        );
        self.end();
    }

    pub fn print_fn_header_info(&mut self, header: hir::FnHeader) {
        self.print_constness(header.constness);

        match header.asyncness {
            hir::IsAsync::NotAsync => {}
            hir::IsAsync::Async => self.word_nbsp("async"),
        }

        self.print_unsafety(header.unsafety);

        if header.abi != Abi::Rust {
            self.word_nbsp("extern");
            self.word_nbsp(header.abi.to_string());
        }

        self.word("fn")
    }

    pub fn print_constness(&mut self, s: hir::Constness) {
        match s {
            hir::Constness::NotConst => {}
            hir::Constness::Const => self.word_nbsp("const"),
        }
    }

    pub fn print_unsafety(&mut self, s: hir::Unsafety) {
        match s {
            hir::Unsafety::Normal => {}
            hir::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }

    pub fn print_is_auto(&mut self, s: hir::IsAuto) {
        match s {
            hir::IsAuto::Yes => self.word_nbsp("auto"),
            hir::IsAuto::No => {}
        }
    }
}

/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
//
// Duplicated from `parse::classify`, but adapted for the HIR.
fn expr_requires_semi_to_be_stmt(e: &hir::Expr<'_>) -> bool {
    !matches!(
        e.kind,
        hir::ExprKind::If(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Block(..)
            | hir::ExprKind::Loop(..)
    )
}

/// This statement requires a semicolon after it.
/// note that in one case (stmt_semi), we've already
/// seen the semicolon, and thus don't need another.
fn stmt_ends_with_semi(stmt: &hir::StmtKind<'_>) -> bool {
    match *stmt {
        hir::StmtKind::Local(_) => true,
        hir::StmtKind::Item(_) => false,
        hir::StmtKind::Expr(e) => expr_requires_semi_to_be_stmt(e),
        hir::StmtKind::Semi(..) => false,
    }
}

fn bin_op_to_assoc_op(op: hir::BinOpKind) -> AssocOp {
    use crate::hir::BinOpKind::*;
    match op {
        Add => AssocOp::Add,
        Sub => AssocOp::Subtract,
        Mul => AssocOp::Multiply,
        Div => AssocOp::Divide,
        Rem => AssocOp::Modulus,

        And => AssocOp::LAnd,
        Or => AssocOp::LOr,

        BitXor => AssocOp::BitXor,
        BitAnd => AssocOp::BitAnd,
        BitOr => AssocOp::BitOr,
        Shl => AssocOp::ShiftLeft,
        Shr => AssocOp::ShiftRight,

        Eq => AssocOp::Equal,
        Lt => AssocOp::Less,
        Le => AssocOp::LessEqual,
        Ne => AssocOp::NotEqual,
        Ge => AssocOp::GreaterEqual,
        Gt => AssocOp::Greater,
    }
}

/// Expressions that syntactically contain an "exterior" struct literal, i.e., not surrounded by any
/// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
fn contains_exterior_struct_lit(value: &hir::Expr<'_>) -> bool {
    match value.kind {
        hir::ExprKind::Struct(..) => true,

        hir::ExprKind::Assign(lhs, rhs, _)
        | hir::ExprKind::AssignOp(_, lhs, rhs)
        | hir::ExprKind::Binary(_, lhs, rhs) => {
            // `X { y: 1 } + X { y: 2 }`
            contains_exterior_struct_lit(lhs) || contains_exterior_struct_lit(rhs)
        }
        hir::ExprKind::Unary(_, x)
        | hir::ExprKind::Cast(x, _)
        | hir::ExprKind::Type(x, _)
        | hir::ExprKind::Field(x, _)
        | hir::ExprKind::Index(x, _) => {
            // `&X { y: 1 }, X { y: 1 }.y`
            contains_exterior_struct_lit(x)
        }

        hir::ExprKind::MethodCall(_, receiver, ..) => {
            // `X { y: 1 }.bar(...)`
            contains_exterior_struct_lit(receiver)
        }

        _ => false,
    }
}
