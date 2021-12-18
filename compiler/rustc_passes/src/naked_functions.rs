//! Checks validity of naked functions.

use rustc_ast::{Attribute, InlineAsmOptions};
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{ErasedMap, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{ExprKind, HirId, InlineAsmOperand, StmtKind};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNDEFINED_NAKED_FUNCTION_ABI;
use rustc_session::lint::builtin::UNSUPPORTED_NAKED_FUNCTIONS;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

fn check_mod_naked_functions(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CheckNakedFunctions { tcx }.as_deep_visitor(),
    );
}

crate fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_naked_functions, ..*providers };
}

struct CheckNakedFunctions<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for CheckNakedFunctions<'tcx> {
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_fn(
        &mut self,
        fk: FnKind<'_>,
        _fd: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        span: Span,
        hir_id: HirId,
    ) {
        let ident_span;
        let fn_header;

        match fk {
            FnKind::Closure => {
                // Closures with a naked attribute are rejected during attribute
                // check. Don't validate them any further.
                return;
            }
            FnKind::ItemFn(ident, _, ref header, ..) => {
                ident_span = ident.span;
                fn_header = header;
            }

            FnKind::Method(ident, ref sig, ..) => {
                ident_span = ident.span;
                fn_header = &sig.header;
            }
        }

        let attrs = self.tcx.hir().attrs(hir_id);
        let naked = attrs.iter().any(|attr| attr.has_name(sym::naked));
        if naked {
            let body = self.tcx.hir().body(body_id);
            check_abi(self.tcx, hir_id, fn_header.abi, ident_span);
            check_no_patterns(self.tcx, body.params);
            check_no_parameters_use(self.tcx, body);
            check_asm(self.tcx, hir_id, body, span);
            check_inline(self.tcx, hir_id, attrs);
        }
    }
}

/// Check that the function isn't inlined.
fn check_inline(tcx: TyCtxt<'_>, hir_id: HirId, attrs: &[Attribute]) {
    for attr in attrs.iter().filter(|attr| attr.has_name(sym::inline)) {
        tcx.struct_span_lint_hir(UNSUPPORTED_NAKED_FUNCTIONS, hir_id, attr.span, |lint| {
            lint.build("naked functions cannot be inlined").emit();
        });
    }
}

/// Checks that function uses non-Rust ABI.
fn check_abi(tcx: TyCtxt<'_>, hir_id: HirId, abi: Abi, fn_ident_span: Span) {
    if abi == Abi::Rust {
        tcx.struct_span_lint_hir(UNDEFINED_NAKED_FUNCTION_ABI, hir_id, fn_ident_span, |lint| {
            lint.build("Rust ABI is unsupported in naked functions").emit();
        });
    }
}

/// Checks that parameters don't use patterns. Mirrors the checks for function declarations.
fn check_no_patterns(tcx: TyCtxt<'_>, params: &[hir::Param<'_>]) {
    for param in params {
        match param.pat.kind {
            hir::PatKind::Wild
            | hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, _, None) => {}
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
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

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
fn check_asm<'tcx>(tcx: TyCtxt<'tcx>, hir_id: HirId, body: &'tcx hir::Body<'tcx>, fn_span: Span) {
    let mut this = CheckInlineAssembly { tcx, items: Vec::new() };
    this.visit_body(body);
    if let [(ItemKind::Asm, _)] = this.items[..] {
        // Ok.
    } else {
        tcx.struct_span_lint_hir(UNSUPPORTED_NAKED_FUNCTIONS, hir_id, fn_span, |lint| {
            let mut diag = lint.build("naked functions must contain a single asm block");
            let mut has_asm = false;
            for &(kind, span) in &this.items {
                match kind {
                    ItemKind::Asm if has_asm => {
                        diag.span_label(
                            span,
                            "multiple asm blocks are unsupported in naked functions",
                        );
                    }
                    ItemKind::Asm => has_asm = true,
                    ItemKind::NonAsm => {
                        diag.span_label(span, "non-asm is unsupported in naked functions");
                    }
                }
            }
            diag.emit();
        });
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
            | ExprKind::Closure(..)
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
                self.check_inline_asm(expr.hir_id, asm, span);
            }

            ExprKind::LlvmInlineAsm(..) => {
                self.items.push((ItemKind::Asm, span));
                self.tcx.struct_span_lint_hir(
                    UNSUPPORTED_NAKED_FUNCTIONS,
                    expr.hir_id,
                    span,
                    |lint| {
                        lint.build(
                            "the LLVM-style inline assembly is unsupported in naked functions",
                        )
                        .help("use the new asm! syntax specified in RFC 2873")
                        .emit();
                    },
                );
            }

            ExprKind::DropTemps(..) | ExprKind::Block(..) | ExprKind::Err => {
                hir::intravisit::walk_expr(self, expr);
            }
        }
    }

    fn check_inline_asm(&self, hir_id: HirId, asm: &'tcx hir::InlineAsm<'tcx>, span: Span) {
        let unsupported_operands: Vec<Span> = asm
            .operands
            .iter()
            .filter_map(|&(ref op, op_sp)| match op {
                InlineAsmOperand::Const { .. } | InlineAsmOperand::Sym { .. } => None,
                InlineAsmOperand::In { .. }
                | InlineAsmOperand::Out { .. }
                | InlineAsmOperand::InOut { .. }
                | InlineAsmOperand::SplitInOut { .. } => Some(op_sp),
            })
            .collect();
        if !unsupported_operands.is_empty() {
            self.tcx.struct_span_lint_hir(
                UNSUPPORTED_NAKED_FUNCTIONS,
                hir_id,
                unsupported_operands,
                |lint| {
                    lint.build("only `const` and `sym` operands are supported in naked functions")
                        .emit();
                },
            );
        }

        let unsupported_options: Vec<&'static str> = [
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
            self.tcx.struct_span_lint_hir(UNSUPPORTED_NAKED_FUNCTIONS, hir_id, span, |lint| {
                lint.build(&format!(
                    "asm options unsupported in naked functions: {}",
                    unsupported_options.join(", ")
                ))
                .emit();
            });
        }

        if !asm.options.contains(InlineAsmOptions::NORETURN) {
            self.tcx.struct_span_lint_hir(UNSUPPORTED_NAKED_FUNCTIONS, hir_id, span, |lint| {
                lint.build("asm in naked functions must use `noreturn` option").emit();
            });
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckInlineAssembly<'tcx> {
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

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
