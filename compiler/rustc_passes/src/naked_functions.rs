//! Checks validity of naked functions.

use rustc_ast::InlineAsmOptions;
use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{ExprKind, InlineAsmOperand, StmtKind};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNDEFINED_NAKED_FUNCTION_ABI;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_naked_functions, ..*providers };
}

fn check_mod_naked_functions(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    let items = tcx.hir_module_items(module_def_id);
    for def_id in items.definitions() {
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
            continue;
        }

        let naked = tcx.has_attr(def_id.to_def_id(), sym::naked);
        if !naked {
            continue;
        }

        let (fn_header, body_id) = match tcx.hir().get_by_def_id(def_id) {
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
        check_abi(tcx, def_id, fn_header.abi);
        check_no_patterns(tcx, body.params);
        check_no_parameters_use(tcx, body);
        check_asm(tcx, def_id, body);
        check_inline(tcx, def_id);
    }
}

/// Check that the function isn't inlined.
fn check_inline(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let attrs = tcx.get_attrs(def_id.to_def_id(), sym::inline);
    for attr in attrs {
        tcx.sess.struct_span_err(attr.span, "naked functions cannot be inlined").emit();
    }
}

/// Checks that function uses non-Rust ABI.
fn check_abi(tcx: TyCtxt<'_>, def_id: LocalDefId, abi: Abi) {
    if abi == Abi::Rust {
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let span = tcx.def_span(def_id);
        tcx.struct_span_lint_hir(UNDEFINED_NAKED_FUNCTION_ABI, hir_id, span, |lint| {
            lint.build("Rust ABI is unsupported in naked functions").emit();
        });
    }
}

/// Checks that parameters don't use patterns. Mirrors the checks for function declarations.
fn check_no_patterns(tcx: TyCtxt<'_>, params: &[hir::Param<'_>]) {
    for param in params {
        match param.pat.kind {
            hir::PatKind::Wild
            | hir::PatKind::Binding(hir::BindingAnnotation::NONE, _, _, None) => {}
            _ => {
                tcx.sess
                    .struct_span_err(
                        param.pat.span,
                        "patterns not allowed in naked function parameters",
                    )
                    .emit();
            }
        }
    }
}

/// Checks that function parameters aren't used in the function body.
fn check_no_parameters_use<'tcx>(tcx: TyCtxt<'tcx>, body: &'tcx hir::Body<'tcx>) {
    let mut params = hir::HirIdSet::default();
    for param in body.params {
        param.pat.each_binding(|_binding_mode, hir_id, _span, _ident| {
            params.insert(hir_id);
        });
    }
    CheckParameters { tcx, params }.visit_body(body);
}

struct CheckParameters<'tcx> {
    tcx: TyCtxt<'tcx>,
    params: hir::HirIdSet,
}

impl<'tcx> Visitor<'tcx> for CheckParameters<'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(var_hir_id), .. },
        )) = expr.kind
        {
            if self.params.contains(var_hir_id) {
                self.tcx
                    .sess
                    .struct_span_err(
                        expr.span,
                        "referencing function parameters is not allowed in naked functions",
                    )
                    .help("follow the calling convention in asm block to use parameters")
                    .emit();
                return;
            }
        }
        hir::intravisit::walk_expr(self, expr);
    }
}

/// Checks that function body contains a single inline assembly block.
fn check_asm<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &'tcx hir::Body<'tcx>) {
    let mut this = CheckInlineAssembly { tcx, items: Vec::new() };
    this.visit_body(body);
    if let [(ItemKind::Asm | ItemKind::Err, _)] = this.items[..] {
        // Ok.
    } else {
        let mut diag = struct_span_err!(
            tcx.sess,
            tcx.def_span(def_id),
            E0787,
            "naked functions must contain a single asm block"
        );

        let mut must_show_error = false;
        let mut has_asm = false;
        let mut has_err = false;
        for &(kind, span) in &this.items {
            match kind {
                ItemKind::Asm if has_asm => {
                    must_show_error = true;
                    diag.span_label(span, "multiple asm blocks are unsupported in naked functions");
                }
                ItemKind::Asm => has_asm = true,
                ItemKind::NonAsm => {
                    must_show_error = true;
                    diag.span_label(span, "non-asm is unsupported in naked functions");
                }
                ItemKind::Err => has_err = true,
            }
        }

        // If the naked function only contains a single asm block and a non-zero number of
        // errors, then don't show an additional error. This allows for appending/prepending
        // `compile_error!("...")` statements and reduces error noise.
        if must_show_error || !has_err {
            diag.emit();
        } else {
            diag.cancel();
        }
    }
}

struct CheckInlineAssembly<'tcx> {
    tcx: TyCtxt<'tcx>,
    items: Vec<(ItemKind, Span)>,
}

#[derive(Copy, Clone)]
enum ItemKind {
    Asm,
    NonAsm,
    Err,
}

impl<'tcx> CheckInlineAssembly<'tcx> {
    fn check_expr(&mut self, expr: &'tcx hir::Expr<'tcx>, span: Span) {
        match expr.kind {
            ExprKind::Box(..)
            | ExprKind::ConstBlock(..)
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
            | ExprKind::Struct(..)
            | ExprKind::Repeat(..)
            | ExprKind::Yield(..) => {
                self.items.push((ItemKind::NonAsm, span));
            }

            ExprKind::InlineAsm(ref asm) => {
                self.items.push((ItemKind::Asm, span));
                self.check_inline_asm(asm, span);
            }

            ExprKind::DropTemps(..) | ExprKind::Block(..) => {
                hir::intravisit::walk_expr(self, expr);
            }

            ExprKind::Err => {
                self.items.push((ItemKind::Err, span));
            }
        }
    }

    fn check_inline_asm(&self, asm: &'tcx hir::InlineAsm<'tcx>, span: Span) {
        let unsupported_operands: Vec<Span> = asm
            .operands
            .iter()
            .filter_map(|&(ref op, op_sp)| match op {
                InlineAsmOperand::Const { .. }
                | InlineAsmOperand::SymFn { .. }
                | InlineAsmOperand::SymStatic { .. } => None,
                InlineAsmOperand::In { .. }
                | InlineAsmOperand::Out { .. }
                | InlineAsmOperand::InOut { .. }
                | InlineAsmOperand::SplitInOut { .. } => Some(op_sp),
            })
            .collect();
        if !unsupported_operands.is_empty() {
            struct_span_err!(
                self.tcx.sess,
                unsupported_operands,
                E0787,
                "only `const` and `sym` operands are supported in naked functions",
            )
            .emit();
        }

        let unsupported_options: Vec<&'static str> = [
            (InlineAsmOptions::MAY_UNWIND, "`may_unwind`"),
            (InlineAsmOptions::NOMEM, "`nomem`"),
            (InlineAsmOptions::NOSTACK, "`nostack`"),
            (InlineAsmOptions::PRESERVES_FLAGS, "`preserves_flags`"),
            (InlineAsmOptions::PURE, "`pure`"),
            (InlineAsmOptions::READONLY, "`readonly`"),
        ]
        .iter()
        .filter_map(|&(option, name)| if asm.options.contains(option) { Some(name) } else { None })
        .collect();

        if !unsupported_options.is_empty() {
            struct_span_err!(
                self.tcx.sess,
                span,
                E0787,
                "asm options unsupported in naked functions: {}",
                unsupported_options.join(", ")
            )
            .emit();
        }

        if !asm.options.contains(InlineAsmOptions::NORETURN) {
            let last_span = asm
                .operands
                .last()
                .map_or_else(|| asm.template_strs.last().unwrap().2, |op| op.1)
                .shrink_to_hi();

            struct_span_err!(
                self.tcx.sess,
                span,
                E0787,
                "asm in naked functions must use `noreturn` option"
            )
            .span_suggestion(
                last_span,
                "consider specifying that the asm block is responsible \
                for returning from the function",
                ", options(noreturn)",
                Applicability::MachineApplicable,
            )
            .emit();
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckInlineAssembly<'tcx> {
    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        match stmt.kind {
            StmtKind::Item(..) => {}
            StmtKind::Local(..) => {
                self.items.push((ItemKind::NonAsm, stmt.span));
            }
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => {
                self.check_expr(expr, stmt.span);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        self.check_expr(&expr, expr.span);
    }
}
