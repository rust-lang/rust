use super::{
    AdtExpr, Arm, Block, ClosureExpr, Expr, ExprKind, InlineAsmExpr, InlineAsmOperand, Pat,
    PatKind, Stmt, StmtKind, Thir,
};

pub trait Visitor<'thir, 'tcx: 'thir>: Sized {
    fn thir(&self) -> &'thir Thir<'tcx>;

    fn visit_expr(&mut self, expr: &'thir Expr<'tcx>) {
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'thir Stmt<'tcx>) {
        walk_stmt(self, stmt);
    }

    fn visit_block(&mut self, block: &'thir Block) {
        walk_block(self, block);
    }

    fn visit_arm(&mut self, arm: &'thir Arm<'tcx>) {
        walk_arm(self, arm);
    }

    fn visit_pat(&mut self, pat: &'thir Pat<'tcx>) {
        walk_pat(self, pat);
    }

    // Note: We don't have visitors for `ty::Const` and `mir::Const`
    // (even though these types occur in THIR) for consistency and to reduce confusion,
    // since the lazy creation of constants during thir construction causes most
    // 'constants' to not be of type `ty::Const` or `mir::Const` at that
    // stage (they are mostly still identified by `DefId` or `hir::Lit`, see
    // the variants `Literal`, `NonHirLiteral` and `NamedConst` in `thir::ExprKind`).
    // You have to manually visit `ty::Const` and `mir::Const` through the
    // other `visit*` functions.
}

pub fn walk_expr<'thir, 'tcx: 'thir, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    expr: &'thir Expr<'tcx>,
) {
    use ExprKind::*;
    match expr.kind {
        Scope { value, region_scope: _, lint_level: _ } => {
            visitor.visit_expr(&visitor.thir()[value])
        }
        Box { value } => visitor.visit_expr(&visitor.thir()[value]),
        If { cond, then, else_opt, if_then_scope: _ } => {
            visitor.visit_expr(&visitor.thir()[cond]);
            visitor.visit_expr(&visitor.thir()[then]);
            if let Some(else_expr) = else_opt {
                visitor.visit_expr(&visitor.thir()[else_expr]);
            }
        }
        Call { fun, ref args, ty: _, from_hir_call: _, fn_span: _ } => {
            visitor.visit_expr(&visitor.thir()[fun]);
            for &arg in &**args {
                visitor.visit_expr(&visitor.thir()[arg]);
            }
        }
        Deref { arg } => visitor.visit_expr(&visitor.thir()[arg]),
        Binary { lhs, rhs, op: _ } | LogicalOp { lhs, rhs, op: _ } => {
            visitor.visit_expr(&visitor.thir()[lhs]);
            visitor.visit_expr(&visitor.thir()[rhs]);
        }
        Unary { arg, op: _ } => visitor.visit_expr(&visitor.thir()[arg]),
        Cast { source } => visitor.visit_expr(&visitor.thir()[source]),
        Use { source } => visitor.visit_expr(&visitor.thir()[source]),
        NeverToAny { source } => visitor.visit_expr(&visitor.thir()[source]),
        PointerCoercion { source, cast: _, is_from_as_cast: _ } => {
            visitor.visit_expr(&visitor.thir()[source])
        }
        Let { expr, ref pat } => {
            visitor.visit_expr(&visitor.thir()[expr]);
            visitor.visit_pat(pat);
        }
        Loop { body } => visitor.visit_expr(&visitor.thir()[body]),
        Match { scrutinee, ref arms, .. } => {
            visitor.visit_expr(&visitor.thir()[scrutinee]);
            for &arm in &**arms {
                visitor.visit_arm(&visitor.thir()[arm]);
            }
        }
        Block { block } => visitor.visit_block(&visitor.thir()[block]),
        Assign { lhs, rhs } | AssignOp { lhs, rhs, op: _ } => {
            visitor.visit_expr(&visitor.thir()[lhs]);
            visitor.visit_expr(&visitor.thir()[rhs]);
        }
        Field { lhs, variant_index: _, name: _ } => visitor.visit_expr(&visitor.thir()[lhs]),
        Index { lhs, index } => {
            visitor.visit_expr(&visitor.thir()[lhs]);
            visitor.visit_expr(&visitor.thir()[index]);
        }
        VarRef { id: _ } | UpvarRef { closure_def_id: _, var_hir_id: _ } => {}
        Borrow { arg, borrow_kind: _ } => visitor.visit_expr(&visitor.thir()[arg]),
        RawBorrow { arg, mutability: _ } => visitor.visit_expr(&visitor.thir()[arg]),
        Break { value, label: _ } => {
            if let Some(value) = value {
                visitor.visit_expr(&visitor.thir()[value])
            }
        }
        Continue { label: _ } => {}
        Return { value } => {
            if let Some(value) = value {
                visitor.visit_expr(&visitor.thir()[value])
            }
        }
        Become { value } => visitor.visit_expr(&visitor.thir()[value]),
        ConstBlock { did: _, args: _ } => {}
        Repeat { value, count: _ } => {
            visitor.visit_expr(&visitor.thir()[value]);
        }
        Array { ref fields } | Tuple { ref fields } => {
            for &field in &**fields {
                visitor.visit_expr(&visitor.thir()[field]);
            }
        }
        Adt(box AdtExpr {
            ref fields,
            ref base,
            adt_def: _,
            variant_index: _,
            args: _,
            user_ty: _,
        }) => {
            for field in &**fields {
                visitor.visit_expr(&visitor.thir()[field.expr]);
            }
            if let Some(base) = base {
                visitor.visit_expr(&visitor.thir()[base.base]);
            }
        }
        PlaceTypeAscription { source, user_ty: _, user_ty_span: _ }
        | ValueTypeAscription { source, user_ty: _, user_ty_span: _ } => {
            visitor.visit_expr(&visitor.thir()[source])
        }
        Closure(box ClosureExpr {
            closure_id: _,
            args: _,
            upvars: _,
            movability: _,
            fake_reads: _,
        }) => {}
        Literal { lit: _, neg: _ } => {}
        NonHirLiteral { lit: _, user_ty: _ } => {}
        ZstLiteral { user_ty: _ } => {}
        NamedConst { def_id: _, args: _, user_ty: _ } => {}
        ConstParam { param: _, def_id: _ } => {}
        StaticRef { alloc_id: _, ty: _, def_id: _ } => {}
        InlineAsm(box InlineAsmExpr {
            asm_macro: _,
            ref operands,
            template: _,
            options: _,
            line_spans: _,
        }) => {
            for op in &**operands {
                use InlineAsmOperand::*;
                match op {
                    In { expr, reg: _ }
                    | Out { expr: Some(expr), reg: _, late: _ }
                    | InOut { expr, reg: _, late: _ } => visitor.visit_expr(&visitor.thir()[*expr]),
                    SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                        visitor.visit_expr(&visitor.thir()[*in_expr]);
                        if let Some(out_expr) = out_expr {
                            visitor.visit_expr(&visitor.thir()[*out_expr]);
                        }
                    }
                    Out { expr: None, reg: _, late: _ }
                    | Const { value: _, span: _ }
                    | SymFn { value: _, span: _ }
                    | SymStatic { def_id: _ } => {}
                    Label { block } => visitor.visit_block(&visitor.thir()[*block]),
                }
            }
        }
        OffsetOf { container: _, fields: _ } => {}
        ThreadLocalRef(_) => {}
        Yield { value } => visitor.visit_expr(&visitor.thir()[value]),
    }
}

pub fn walk_stmt<'thir, 'tcx: 'thir, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    stmt: &'thir Stmt<'tcx>,
) {
    match &stmt.kind {
        StmtKind::Expr { expr, scope: _ } => visitor.visit_expr(&visitor.thir()[*expr]),
        StmtKind::Let {
            initializer,
            remainder_scope: _,
            init_scope: _,
            ref pattern,
            lint_level: _,
            else_block,
            span: _,
        } => {
            if let Some(init) = initializer {
                visitor.visit_expr(&visitor.thir()[*init]);
            }
            visitor.visit_pat(pattern);
            if let Some(block) = else_block {
                visitor.visit_block(&visitor.thir()[*block])
            }
        }
    }
}

pub fn walk_block<'thir, 'tcx: 'thir, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    block: &'thir Block,
) {
    for &stmt in &*block.stmts {
        visitor.visit_stmt(&visitor.thir()[stmt]);
    }
    if let Some(expr) = block.expr {
        visitor.visit_expr(&visitor.thir()[expr]);
    }
}

pub fn walk_arm<'thir, 'tcx: 'thir, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    arm: &'thir Arm<'tcx>,
) {
    if let Some(expr) = arm.guard {
        visitor.visit_expr(&visitor.thir()[expr])
    }
    visitor.visit_pat(&arm.pattern);
    visitor.visit_expr(&visitor.thir()[arm.body]);
}

pub fn walk_pat<'thir, 'tcx: 'thir, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    pat: &'thir Pat<'tcx>,
) {
    use PatKind::*;
    match &pat.kind {
        AscribeUserType { subpattern, ascription: _ }
        | Deref { subpattern }
        | DerefPattern { subpattern, .. }
        | Binding { subpattern: Some(subpattern), .. } => visitor.visit_pat(subpattern),
        Binding { .. } | Wild | Never | Error(_) => {}
        Variant { subpatterns, adt_def: _, args: _, variant_index: _ } | Leaf { subpatterns } => {
            for subpattern in subpatterns {
                visitor.visit_pat(&subpattern.pattern);
            }
        }
        Constant { value: _ } => {}
        ExpandedConstant { def_id: _, is_inline: _, subpattern } => visitor.visit_pat(subpattern),
        Range(_) => {}
        Slice { prefix, slice, suffix } | Array { prefix, slice, suffix } => {
            for subpattern in prefix.iter() {
                visitor.visit_pat(subpattern);
            }
            if let Some(pat) = slice {
                visitor.visit_pat(pat);
            }
            for subpattern in suffix.iter() {
                visitor.visit_pat(subpattern);
            }
        }
        Or { pats } => {
            for pat in pats.iter() {
                visitor.visit_pat(pat);
            }
        }
    };
}
