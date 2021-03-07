use crate::thir::*;

pub trait Visitor<'thir, 'tcx>: Sized {
    fn visit_expr(&mut self, expr: &'thir Expr<'thir, 'tcx>) {
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'thir Stmt<'thir, 'tcx>) {
        walk_stmt(self, stmt);
    }

    fn visit_block(&mut self, block: &Block<'thir, 'tcx>) {
        walk_block(self, block);
    }

    fn visit_arm(&mut self, arm: &'thir Arm<'thir, 'tcx>) {
        walk_arm(self, arm);
    }

    fn visit_const(&mut self, _cnst: &'tcx Const<'tcx>) {}
}

pub fn walk_expr<'thir, 'tcx, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    expr: &'thir Expr<'thir, 'tcx>,
) {
    use ExprKind::*;
    match expr.kind {
        Scope { value, .. } => visitor.visit_expr(value),
        Box { value } => visitor.visit_expr(value),
        If { cond, then, else_opt } => {
            visitor.visit_expr(cond);
            visitor.visit_expr(then);
            if let Some(else_expr) = else_opt {
                visitor.visit_expr(else_expr);
            }
        }
        Call { fun, args, .. } => {
            visitor.visit_expr(fun);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        Deref { arg } => visitor.visit_expr(arg),
        Binary { lhs, rhs, .. } | LogicalOp { lhs, rhs, .. } => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        Unary { arg, .. } => visitor.visit_expr(arg),
        Cast { source } => visitor.visit_expr(source),
        Use { source } => visitor.visit_expr(source),
        NeverToAny { source } => visitor.visit_expr(source),
        Pointer { source, .. } => visitor.visit_expr(source),
        Loop { body } => visitor.visit_expr(body),
        Match { scrutinee, arms } => {
            visitor.visit_expr(scrutinee);
            for arm in arms {
                visitor.visit_arm(arm);
            }
        }
        Block { ref body } => visitor.visit_block(body),
        Assign { lhs, rhs } | AssignOp { lhs, rhs, .. } => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        Field { lhs, .. } => visitor.visit_expr(lhs),
        Index { lhs, index } => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(index);
        }
        VarRef { .. } | UpvarRef { .. } => {}
        Borrow { arg, .. } => visitor.visit_expr(arg),
        AddressOf { arg, .. } => visitor.visit_expr(arg),
        Break { value, .. } => {
            if let Some(value) = value {
                visitor.visit_expr(value)
            }
        }
        Continue { .. } => {}
        Return { value } => {
            if let Some(value) = value {
                visitor.visit_expr(value)
            }
        }
        ConstBlock { value } => visitor.visit_const(value),
        Repeat { value, count } => {
            visitor.visit_expr(value);
            visitor.visit_const(count);
        }
        Array { fields } | Tuple { fields } => {
            for field in fields {
                visitor.visit_expr(field);
            }
        }
        Adt { fields, ref base, .. } => {
            for field in fields {
                visitor.visit_expr(field.expr);
            }
            if let Some(base) = base {
                visitor.visit_expr(base.base);
            }
        }
        PlaceTypeAscription { source, .. } | ValueTypeAscription { source, .. } => {
            visitor.visit_expr(source)
        }
        Closure { .. } => {}
        Literal { literal, .. } => visitor.visit_const(literal),
        StaticRef { literal, .. } => visitor.visit_const(literal),
        InlineAsm { operands, .. } => {
            for op in operands {
                use InlineAsmOperand::*;
                match op {
                    In { expr, .. }
                    | Out { expr: Some(expr), .. }
                    | InOut { expr, .. }
                    | SymFn { expr } => visitor.visit_expr(expr),
                    SplitInOut { in_expr, out_expr, .. } => {
                        visitor.visit_expr(in_expr);
                        if let Some(out_expr) = out_expr {
                            visitor.visit_expr(out_expr);
                        }
                    }
                    Out { .. } | Const { .. } | SymStatic { .. } => {}
                }
            }
        }
        ThreadLocalRef(_) => {}
        LlvmInlineAsm { outputs, inputs, .. } => {
            for out_expr in outputs {
                visitor.visit_expr(out_expr);
            }
            for in_expr in inputs {
                visitor.visit_expr(in_expr);
            }
        }
        Yield { value } => visitor.visit_expr(value),
    }
}

pub fn walk_stmt<'thir, 'tcx, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    stmt: &'thir Stmt<'thir, 'tcx>,
) {
    match stmt.kind {
        StmtKind::Expr { expr, .. } => visitor.visit_expr(expr),
        StmtKind::Let { pattern: _, initializer, .. } => {
            if let Some(init) = initializer {
                visitor.visit_expr(init);
            }
        }
    }
}

pub fn walk_block<'thir, 'tcx, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    block: &Block<'thir, 'tcx>,
) {
    for stmt in block.stmts {
        visitor.visit_stmt(stmt);
    }
    if let Some(expr) = block.expr {
        visitor.visit_expr(expr);
    }
}

pub fn walk_arm<'thir, 'tcx, V: Visitor<'thir, 'tcx>>(
    visitor: &mut V,
    arm: &'thir Arm<'thir, 'tcx>,
) {
    match arm.guard {
        Some(Guard::If(expr)) => visitor.visit_expr(expr),
        Some(Guard::IfLet(ref _pat, expr)) => {
            visitor.visit_expr(expr);
        }
        None => {}
    }
    visitor.visit_expr(arm.body);
}
