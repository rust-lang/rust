//! Checks validity of naked functions.

use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{ExprKind, HirIdSet, StmtKind};
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNDEFINED_NAKED_FUNCTION_ABI;
use rustc_span::Span;
use rustc_span::symbol::sym;
use rustc_target::spec::abi::Abi;

use crate::errors::{
    NakedAsmOutsideNakedFn, NakedFunctionsAsmBlock, NakedFunctionsMustNakedAsm, NoPatterns,
    ParamsNotAllowed, UndefinedNakedFunctionAbi,
};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_naked_functions, ..*providers };
}

fn check_mod_naked_functions(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    let items = tcx.hir_module_items(module_def_id);
    for def_id in items.definitions() {
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
            continue;
        }

        let (fn_header, body_id) = match tcx.hir_node_by_def_id(def_id) {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, _, body_id), .. })
            | hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, hir::TraitFn::Provided(body_id)),
                ..
            })
            | hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, body_id),
                ..
            }) => (sig.header, *body_id),
            _ => continue,
        };

        let body = tcx.hir().body(body_id);

        if tcx.has_attr(def_id, sym::naked) {
            check_abi(tcx, def_id, fn_header.abi);
            check_no_patterns(tcx, body.params);
            check_no_parameters_use(tcx, body);
            check_asm(tcx, def_id, body);
        } else {
            // `naked_asm!` is not allowed outside of functions marked as `#[naked]`
            let mut visitor = CheckNakedAsmInNakedFn { tcx };
            visitor.visit_body(body);
        }
    }
}

/// Checks that function uses non-Rust ABI.
fn check_abi(tcx: TyCtxt<'_>, def_id: LocalDefId, abi: Abi) {
    if abi == Abi::Rust {
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        let span = tcx.def_span(def_id);
        tcx.emit_node_span_lint(
            UNDEFINED_NAKED_FUNCTION_ABI,
            hir_id,
            span,
            UndefinedNakedFunctionAbi,
        );
    }
}

/// Checks that parameters don't use patterns. Mirrors the checks for function declarations.
fn check_no_patterns(tcx: TyCtxt<'_>, params: &[hir::Param<'_>]) {
    for param in params {
        match param.pat.kind {
            hir::PatKind::Wild | hir::PatKind::Binding(hir::BindingMode::NONE, _, _, None) => {}
            _ => {
                tcx.dcx().emit_err(NoPatterns { span: param.pat.span });
            }
        }
    }
}

/// Checks that function parameters aren't used in the function body.
fn check_no_parameters_use<'tcx>(tcx: TyCtxt<'tcx>, body: &'tcx hir::Body<'tcx>) {
    let mut params = HirIdSet::default();
    for param in body.params {
        param.pat.each_binding(|_binding_mode, hir_id, _span, _ident| {
            params.insert(hir_id);
        });
    }
    CheckParameters { tcx, params }.visit_body(body);
}

struct CheckParameters<'tcx> {
    tcx: TyCtxt<'tcx>,
    params: HirIdSet,
}

impl<'tcx> Visitor<'tcx> for CheckParameters<'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(var_hir_id), .. },
        )) = expr.kind
        {
            if self.params.contains(var_hir_id) {
                self.tcx.dcx().emit_err(ParamsNotAllowed { span: expr.span });
                return;
            }
        }
        hir::intravisit::walk_expr(self, expr);
    }
}

/// Checks that function body contains a single inline assembly block.
fn check_asm<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &'tcx hir::Body<'tcx>) {
    let mut this = CheckInlineAssembly { items: Vec::new() };
    this.visit_body(body);
    if let [(ItemKind::NakedAsm | ItemKind::Err, _)] = this.items[..] {
        // Ok.
    } else {
        let mut must_show_error = false;
        let mut has_naked_asm = false;
        let mut has_err = false;
        let mut multiple_asms = vec![];
        let mut non_asms = vec![];
        for &(kind, span) in &this.items {
            match kind {
                ItemKind::NakedAsm if has_naked_asm => {
                    must_show_error = true;
                    multiple_asms.push(span);
                }
                ItemKind::NakedAsm => has_naked_asm = true,
                ItemKind::InlineAsm => {
                    has_err = true;

                    tcx.dcx().emit_err(NakedFunctionsMustNakedAsm { span });
                }
                ItemKind::NonAsm => {
                    must_show_error = true;
                    non_asms.push(span);
                }
                ItemKind::Err => has_err = true,
            }
        }

        // If the naked function only contains a single asm block and a non-zero number of
        // errors, then don't show an additional error. This allows for appending/prepending
        // `compile_error!("...")` statements and reduces error noise.
        if must_show_error || !has_err {
            tcx.dcx().emit_err(NakedFunctionsAsmBlock {
                span: tcx.def_span(def_id),
                multiple_asms,
                non_asms,
            });
        }
    }
}

struct CheckInlineAssembly {
    items: Vec<(ItemKind, Span)>,
}

#[derive(Copy, Clone)]
enum ItemKind {
    NakedAsm,
    InlineAsm,
    NonAsm,
    Err,
}

impl CheckInlineAssembly {
    fn check_expr<'tcx>(&mut self, expr: &'tcx hir::Expr<'tcx>, span: Span) {
        match expr.kind {
            ExprKind::ConstBlock(..)
            | ExprKind::Array(..)
            | ExprKind::Call(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Tup(..)
            | ExprKind::Binary(..)
            | ExprKind::Unary(..)
            | ExprKind::Lit(..)
            | ExprKind::Cast(..)
            | ExprKind::Type(..)
            | ExprKind::Loop(..)
            | ExprKind::Match(..)
            | ExprKind::If(..)
            | ExprKind::Closure { .. }
            | ExprKind::Assign(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Field(..)
            | ExprKind::Index(..)
            | ExprKind::Path(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Let(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::OffsetOf(..)
            | ExprKind::Become(..)
            | ExprKind::Struct(..)
            | ExprKind::Repeat(..)
            | ExprKind::Yield(..) => {
                self.items.push((ItemKind::NonAsm, span));
            }

            ExprKind::InlineAsm(asm) => match asm.asm_macro {
                rustc_ast::AsmMacro::Asm => {
                    self.items.push((ItemKind::InlineAsm, span));
                }
                rustc_ast::AsmMacro::NakedAsm => {
                    self.items.push((ItemKind::NakedAsm, span));
                }
                rustc_ast::AsmMacro::GlobalAsm => {
                    span_bug!(span, "`global_asm!` is not allowed in this position")
                }
            },

            ExprKind::DropTemps(..) | ExprKind::Block(..) => {
                hir::intravisit::walk_expr(self, expr);
            }

            ExprKind::Err(_) => {
                self.items.push((ItemKind::Err, span));
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckInlineAssembly {
    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        match stmt.kind {
            StmtKind::Item(..) => {}
            StmtKind::Let(..) => {
                self.items.push((ItemKind::NonAsm, stmt.span));
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.check_expr(expr, stmt.span);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        self.check_expr(expr, expr.span);
    }
}

struct CheckNakedAsmInNakedFn<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for CheckNakedAsmInNakedFn<'tcx> {
    type NestedFilter = OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let ExprKind::InlineAsm(inline_asm) = expr.kind {
            if let rustc_ast::AsmMacro::NakedAsm = inline_asm.asm_macro {
                self.tcx.dcx().emit_err(NakedAsmOutsideNakedFn { span: expr.span });
            }
        }

        hir::intravisit::walk_expr(self, expr);
    }
}
