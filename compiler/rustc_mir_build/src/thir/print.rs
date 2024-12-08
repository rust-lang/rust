use std::fmt::{self, Write};

use rustc_middle::query::TyCtxtAt;
use rustc_middle::thir::*;
use rustc_middle::ty;
use rustc_span::def_id::LocalDefId;

pub(crate) fn thir_tree(tcx: TyCtxtAt<'_>, owner_def: LocalDefId) -> String {
    match super::cx::thir_body(*tcx, owner_def) {
        Ok((thir, _)) => {
            let thir = thir.steal();
            let mut printer = ThirPrinter::new(&thir);
            printer.print();
            printer.into_buffer()
        }
        Err(_) => "error".into(),
    }
}

pub(crate) fn thir_flat(tcx: TyCtxtAt<'_>, owner_def: LocalDefId) -> String {
    match super::cx::thir_body(*tcx, owner_def) {
        Ok((thir, _)) => format!("{:#?}", thir.steal()),
        Err(_) => "error".into(),
    }
}

struct ThirPrinter<'a, 'tcx> {
    thir: &'a Thir<'tcx>,
    fmt: String,
}

const INDENT: &str = "    ";

macro_rules! print_indented {
    ($writer:ident, $s:expr, $indent_lvl:expr) => {
        $writer.indent($indent_lvl);
        writeln!($writer, "{}", $s).expect("unable to write to ThirPrinter");
    };
}

impl<'a, 'tcx> Write for ThirPrinter<'a, 'tcx> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.fmt.push_str(s);
        Ok(())
    }
}

impl<'a, 'tcx> ThirPrinter<'a, 'tcx> {
    fn new(thir: &'a Thir<'tcx>) -> Self {
        Self { thir, fmt: String::new() }
    }

    fn indent(&mut self, level: usize) {
        for _ in 0..level {
            self.fmt.push_str(INDENT);
        }
    }

    fn print(&mut self) {
        print_indented!(self, "params: [", 0);
        for param in self.thir.params.iter() {
            self.print_param(param, 1);
        }
        print_indented!(self, "]", 0);

        print_indented!(self, "body:", 0);
        let expr = ExprId::from_usize(self.thir.exprs.len() - 1);
        self.print_expr(expr, 1);
    }

    fn into_buffer(self) -> String {
        self.fmt
    }

    fn print_param(&mut self, param: &Param<'tcx>, depth_lvl: usize) {
        let Param { pat, ty, ty_span, self_kind, hir_id } = param;

        print_indented!(self, "Param {", depth_lvl);
        print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 1);
        print_indented!(self, format!("ty_span: {:?}", ty_span), depth_lvl + 1);
        print_indented!(self, format!("self_kind: {:?}", self_kind), depth_lvl + 1);
        print_indented!(self, format!("hir_id: {:?}", hir_id), depth_lvl + 1);

        if let Some(pat) = pat {
            print_indented!(self, "param: Some( ", depth_lvl + 1);
            self.print_pat(pat, depth_lvl + 2);
            print_indented!(self, ")", depth_lvl + 1);
        } else {
            print_indented!(self, "param: None", depth_lvl + 1);
        }

        print_indented!(self, "}", depth_lvl);
    }

    fn print_block(&mut self, block_id: BlockId, depth_lvl: usize) {
        let Block { targeted_by_break, span, region_scope, stmts, expr, safety_mode } =
            &self.thir.blocks[block_id];

        print_indented!(self, "Block {", depth_lvl);
        print_indented!(self, format!("targeted_by_break: {}", targeted_by_break), depth_lvl + 1);
        print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
        print_indented!(self, format!("region_scope: {:?}", region_scope), depth_lvl + 1);
        print_indented!(self, format!("safety_mode: {:?}", safety_mode), depth_lvl + 1);

        if stmts.len() > 0 {
            print_indented!(self, "stmts: [", depth_lvl + 1);
            for stmt in stmts.iter() {
                self.print_stmt(*stmt, depth_lvl + 2);
            }
            print_indented!(self, "]", depth_lvl + 1);
        } else {
            print_indented!(self, "stmts: []", depth_lvl + 1);
        }

        if let Some(expr_id) = expr {
            print_indented!(self, "expr:", depth_lvl + 1);
            self.print_expr(*expr_id, depth_lvl + 2);
        } else {
            print_indented!(self, "expr: []", depth_lvl + 1);
        }

        print_indented!(self, "}", depth_lvl);
    }

    fn print_stmt(&mut self, stmt_id: StmtId, depth_lvl: usize) {
        let Stmt { kind } = &self.thir.stmts[stmt_id];

        print_indented!(self, "Stmt {", depth_lvl);

        match kind {
            StmtKind::Expr { scope, expr } => {
                print_indented!(self, "kind: Expr {", depth_lvl + 1);
                print_indented!(self, format!("scope: {:?}", scope), depth_lvl + 2);
                print_indented!(self, "expr:", depth_lvl + 2);
                self.print_expr(*expr, depth_lvl + 3);
                print_indented!(self, "}", depth_lvl + 1);
            }
            StmtKind::Let {
                remainder_scope,
                init_scope,
                pattern,
                initializer,
                else_block,
                lint_level,
                span,
            } => {
                print_indented!(self, "kind: Let {", depth_lvl + 1);
                print_indented!(
                    self,
                    format!("remainder_scope: {:?}", remainder_scope),
                    depth_lvl + 2
                );
                print_indented!(self, format!("init_scope: {:?}", init_scope), depth_lvl + 2);

                print_indented!(self, "pattern: ", depth_lvl + 2);
                self.print_pat(pattern, depth_lvl + 3);
                print_indented!(self, ",", depth_lvl + 2);

                if let Some(init) = initializer {
                    print_indented!(self, "initializer: Some(", depth_lvl + 2);
                    self.print_expr(*init, depth_lvl + 3);
                    print_indented!(self, ")", depth_lvl + 2);
                } else {
                    print_indented!(self, "initializer: None", depth_lvl + 2);
                }

                if let Some(else_block) = else_block {
                    print_indented!(self, "else_block: Some(", depth_lvl + 2);
                    self.print_block(*else_block, depth_lvl + 3);
                    print_indented!(self, ")", depth_lvl + 2);
                } else {
                    print_indented!(self, "else_block: None", depth_lvl + 2);
                }

                print_indented!(self, format!("lint_level: {:?}", lint_level), depth_lvl + 2);
                print_indented!(self, format!("span: {:?}", span), depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
        }

        print_indented!(self, "}", depth_lvl);
    }

    fn print_expr(&mut self, expr: ExprId, depth_lvl: usize) {
        let Expr { ty, temp_lifetime, span, kind } = &self.thir[expr];
        print_indented!(self, "Expr {", depth_lvl);
        print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 1);
        print_indented!(self, format!("temp_lifetime: {:?}", temp_lifetime), depth_lvl + 1);
        print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
        print_indented!(self, "kind: ", depth_lvl + 1);
        self.print_expr_kind(kind, depth_lvl + 2);
        print_indented!(self, "}", depth_lvl);
    }

    fn print_expr_kind(&mut self, expr_kind: &ExprKind<'tcx>, depth_lvl: usize) {
        use rustc_middle::thir::ExprKind::*;

        match expr_kind {
            Scope { region_scope, value, lint_level } => {
                print_indented!(self, "Scope {", depth_lvl);
                print_indented!(self, format!("region_scope: {:?}", region_scope), depth_lvl + 1);
                print_indented!(self, format!("lint_level: {:?}", lint_level), depth_lvl + 1);
                print_indented!(self, "value:", depth_lvl + 1);
                self.print_expr(*value, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Box { value } => {
                print_indented!(self, "Box {", depth_lvl);
                self.print_expr(*value, depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            If { if_then_scope, cond, then, else_opt } => {
                print_indented!(self, "If {", depth_lvl);
                print_indented!(self, format!("if_then_scope: {:?}", if_then_scope), depth_lvl + 1);
                print_indented!(self, "cond:", depth_lvl + 1);
                self.print_expr(*cond, depth_lvl + 2);
                print_indented!(self, "then:", depth_lvl + 1);
                self.print_expr(*then, depth_lvl + 2);

                if let Some(else_expr) = else_opt {
                    print_indented!(self, "else:", depth_lvl + 1);
                    self.print_expr(*else_expr, depth_lvl + 2);
                }

                print_indented!(self, "}", depth_lvl);
            }
            Call { fun, args, ty, from_hir_call, fn_span } => {
                print_indented!(self, "Call {", depth_lvl);
                print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 1);
                print_indented!(self, format!("from_hir_call: {}", from_hir_call), depth_lvl + 1);
                print_indented!(self, format!("fn_span: {:?}", fn_span), depth_lvl + 1);
                print_indented!(self, "fun:", depth_lvl + 1);
                self.print_expr(*fun, depth_lvl + 2);

                if args.len() > 0 {
                    print_indented!(self, "args: [", depth_lvl + 1);
                    for arg in args.iter() {
                        self.print_expr(*arg, depth_lvl + 2);
                    }
                    print_indented!(self, "]", depth_lvl + 1);
                } else {
                    print_indented!(self, "args: []", depth_lvl + 1);
                }

                print_indented!(self, "}", depth_lvl);
            }
            Deref { arg } => {
                print_indented!(self, "Deref {", depth_lvl);
                self.print_expr(*arg, depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Binary { op, lhs, rhs } => {
                print_indented!(self, "Binary {", depth_lvl);
                print_indented!(self, format!("op: {:?}", op), depth_lvl + 1);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "rhs:", depth_lvl + 1);
                self.print_expr(*rhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            LogicalOp { op, lhs, rhs } => {
                print_indented!(self, "LogicalOp {", depth_lvl);
                print_indented!(self, format!("op: {:?}", op), depth_lvl + 1);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "rhs:", depth_lvl + 1);
                self.print_expr(*rhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Unary { op, arg } => {
                print_indented!(self, "Unary {", depth_lvl);
                print_indented!(self, format!("op: {:?}", op), depth_lvl + 1);
                print_indented!(self, "arg:", depth_lvl + 1);
                self.print_expr(*arg, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Cast { source } => {
                print_indented!(self, "Cast {", depth_lvl);
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Use { source } => {
                print_indented!(self, "Use {", depth_lvl);
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            NeverToAny { source } => {
                print_indented!(self, "NeverToAny {", depth_lvl);
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            PointerCoercion { cast, is_from_as_cast, source } => {
                print_indented!(self, "Pointer {", depth_lvl);
                print_indented!(self, format!("cast: {:?}", cast), depth_lvl + 1);
                print_indented!(
                    self,
                    format!("is_from_as_cast: {:?}", is_from_as_cast),
                    depth_lvl + 1
                );
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Loop { body } => {
                print_indented!(self, "Loop (", depth_lvl);
                print_indented!(self, "body:", depth_lvl + 1);
                self.print_expr(*body, depth_lvl + 2);
                print_indented!(self, ")", depth_lvl);
            }
            Let { expr, pat } => {
                print_indented!(self, "Let {", depth_lvl);
                print_indented!(self, "expr:", depth_lvl + 1);
                self.print_expr(*expr, depth_lvl + 2);
                print_indented!(self, format!("pat: {:?}", pat), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Match { scrutinee, arms, .. } => {
                print_indented!(self, "Match {", depth_lvl);
                print_indented!(self, "scrutinee:", depth_lvl + 1);
                self.print_expr(*scrutinee, depth_lvl + 2);

                print_indented!(self, "arms: [", depth_lvl + 1);
                for arm_id in arms.iter() {
                    self.print_arm(*arm_id, depth_lvl + 2);
                }
                print_indented!(self, "]", depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Block { block } => self.print_block(*block, depth_lvl),
            Assign { lhs, rhs } => {
                print_indented!(self, "Assign {", depth_lvl);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "rhs:", depth_lvl + 1);
                self.print_expr(*rhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            AssignOp { op, lhs, rhs } => {
                print_indented!(self, "AssignOp {", depth_lvl);
                print_indented!(self, format!("op: {:?}", op), depth_lvl + 1);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "rhs:", depth_lvl + 1);
                self.print_expr(*rhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Field { lhs, variant_index, name } => {
                print_indented!(self, "Field {", depth_lvl);
                print_indented!(self, format!("variant_index: {:?}", variant_index), depth_lvl + 1);
                print_indented!(self, format!("name: {:?}", name), depth_lvl + 1);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Index { lhs, index } => {
                print_indented!(self, "Index {", depth_lvl);
                print_indented!(self, format!("index: {:?}", index), depth_lvl + 1);
                print_indented!(self, "lhs:", depth_lvl + 1);
                self.print_expr(*lhs, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            VarRef { id } => {
                print_indented!(self, "VarRef {", depth_lvl);
                print_indented!(self, format!("id: {:?}", id), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            UpvarRef { closure_def_id, var_hir_id } => {
                print_indented!(self, "UpvarRef {", depth_lvl);
                print_indented!(
                    self,
                    format!("closure_def_id: {:?}", closure_def_id),
                    depth_lvl + 1
                );
                print_indented!(self, format!("var_hir_id: {:?}", var_hir_id), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Borrow { borrow_kind, arg } => {
                print_indented!(self, "Borrow (", depth_lvl);
                print_indented!(self, format!("borrow_kind: {:?}", borrow_kind), depth_lvl + 1);
                print_indented!(self, "arg:", depth_lvl + 1);
                self.print_expr(*arg, depth_lvl + 2);
                print_indented!(self, ")", depth_lvl);
            }
            RawBorrow { mutability, arg } => {
                print_indented!(self, "RawBorrow {", depth_lvl);
                print_indented!(self, format!("mutability: {:?}", mutability), depth_lvl + 1);
                print_indented!(self, "arg:", depth_lvl + 1);
                self.print_expr(*arg, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Break { label, value } => {
                print_indented!(self, "Break (", depth_lvl);
                print_indented!(self, format!("label: {:?}", label), depth_lvl + 1);

                if let Some(value) = value {
                    print_indented!(self, "value:", depth_lvl + 1);
                    self.print_expr(*value, depth_lvl + 2);
                }

                print_indented!(self, ")", depth_lvl);
            }
            Continue { label } => {
                print_indented!(self, "Continue {", depth_lvl);
                print_indented!(self, format!("label: {:?}", label), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Return { value } => {
                print_indented!(self, "Return {", depth_lvl);
                print_indented!(self, "value:", depth_lvl + 1);

                if let Some(value) = value {
                    self.print_expr(*value, depth_lvl + 2);
                }

                print_indented!(self, "}", depth_lvl);
            }
            Become { value } => {
                print_indented!(self, "Become {", depth_lvl);
                print_indented!(self, "value:", depth_lvl + 1);
                self.print_expr(*value, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            ConstBlock { did, args } => {
                print_indented!(self, "ConstBlock {", depth_lvl);
                print_indented!(self, format!("did: {:?}", did), depth_lvl + 1);
                print_indented!(self, format!("args: {:?}", args), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Repeat { value, count } => {
                print_indented!(self, "Repeat {", depth_lvl);
                print_indented!(self, format!("count: {:?}", count), depth_lvl + 1);
                print_indented!(self, "value:", depth_lvl + 1);
                self.print_expr(*value, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Array { fields } => {
                print_indented!(self, "Array {", depth_lvl);
                print_indented!(self, "fields: [", depth_lvl + 1);
                for field_id in fields.iter() {
                    self.print_expr(*field_id, depth_lvl + 2);
                }
                print_indented!(self, "]", depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Tuple { fields } => {
                print_indented!(self, "Tuple {", depth_lvl);
                print_indented!(self, "fields: [", depth_lvl + 1);
                for field_id in fields.iter() {
                    self.print_expr(*field_id, depth_lvl + 2);
                }
                print_indented!(self, "]", depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Adt(adt_expr) => {
                print_indented!(self, "Adt {", depth_lvl);
                self.print_adt_expr(&**adt_expr, depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            PlaceTypeAscription { source, user_ty, user_ty_span } => {
                print_indented!(self, "PlaceTypeAscription {", depth_lvl);
                print_indented!(self, format!("user_ty: {:?}", user_ty), depth_lvl + 1);
                print_indented!(self, format!("user_ty_span: {:?}", user_ty_span), depth_lvl + 1);
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            ValueTypeAscription { source, user_ty, user_ty_span } => {
                print_indented!(self, "ValueTypeAscription {", depth_lvl);
                print_indented!(self, format!("user_ty: {:?}", user_ty), depth_lvl + 1);
                print_indented!(self, format!("user_ty_span: {:?}", user_ty_span), depth_lvl + 1);
                print_indented!(self, "source:", depth_lvl + 1);
                self.print_expr(*source, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Closure(closure_expr) => {
                print_indented!(self, "Closure {", depth_lvl);
                print_indented!(self, "closure_expr:", depth_lvl + 1);
                self.print_closure_expr(&**closure_expr, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            Literal { lit, neg } => {
                print_indented!(
                    self,
                    format!("Literal( lit: {:?}, neg: {:?})\n", lit, neg),
                    depth_lvl
                );
            }
            NonHirLiteral { lit, user_ty } => {
                print_indented!(self, "NonHirLiteral {", depth_lvl);
                print_indented!(self, format!("lit: {:?}", lit), depth_lvl + 1);
                print_indented!(self, format!("user_ty: {:?}", user_ty), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            ZstLiteral { user_ty } => {
                print_indented!(self, format!("ZstLiteral(user_ty: {:?})", user_ty), depth_lvl);
            }
            NamedConst { def_id, args, user_ty } => {
                print_indented!(self, "NamedConst {", depth_lvl);
                print_indented!(self, format!("def_id: {:?}", def_id), depth_lvl + 1);
                print_indented!(self, format!("user_ty: {:?}", user_ty), depth_lvl + 1);
                print_indented!(self, format!("args: {:?}", args), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            ConstParam { param, def_id } => {
                print_indented!(self, "ConstParam {", depth_lvl);
                print_indented!(self, format!("def_id: {:?}", def_id), depth_lvl + 1);
                print_indented!(self, format!("param: {:?}", param), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            StaticRef { alloc_id, ty, def_id } => {
                print_indented!(self, "StaticRef {", depth_lvl);
                print_indented!(self, format!("def_id: {:?}", def_id), depth_lvl + 1);
                print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 1);
                print_indented!(self, format!("alloc_id: {:?}", alloc_id), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            InlineAsm(expr) => {
                print_indented!(self, "InlineAsm {", depth_lvl);
                print_indented!(self, "expr:", depth_lvl + 1);
                self.print_inline_asm_expr(&**expr, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
            OffsetOf { container, fields } => {
                print_indented!(self, "OffsetOf {", depth_lvl);
                print_indented!(self, format!("container: {:?}", container), depth_lvl + 1);
                print_indented!(self, "fields: [", depth_lvl + 1);

                for field in fields.iter() {
                    print_indented!(self, format!("{:?}", field), depth_lvl + 2);
                    print_indented!(self, ",", depth_lvl + 1);
                }

                print_indented!(self, "]", depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            ThreadLocalRef(def_id) => {
                print_indented!(self, "ThreadLocalRef {", depth_lvl);
                print_indented!(self, format!("def_id: {:?}", def_id), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl);
            }
            Yield { value } => {
                print_indented!(self, "Yield {", depth_lvl);
                print_indented!(self, "value:", depth_lvl + 1);
                self.print_expr(*value, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl);
            }
        }
    }

    fn print_adt_expr(&mut self, adt_expr: &AdtExpr<'tcx>, depth_lvl: usize) {
        print_indented!(self, "adt_def:", depth_lvl);
        self.print_adt_def(adt_expr.adt_def, depth_lvl + 1);
        print_indented!(
            self,
            format!("variant_index: {:?}", adt_expr.variant_index),
            depth_lvl + 1
        );
        print_indented!(self, format!("args: {:?}", adt_expr.args), depth_lvl + 1);
        print_indented!(self, format!("user_ty: {:?}", adt_expr.user_ty), depth_lvl + 1);

        for (i, field_expr) in adt_expr.fields.iter().enumerate() {
            print_indented!(self, format!("field {}:", i), depth_lvl + 1);
            self.print_expr(field_expr.expr, depth_lvl + 2);
        }

        if let Some(ref base) = adt_expr.base {
            print_indented!(self, "base:", depth_lvl + 1);
            self.print_fru_info(base, depth_lvl + 2);
        } else {
            print_indented!(self, "base: None", depth_lvl + 1);
        }
    }

    fn print_adt_def(&mut self, adt_def: ty::AdtDef<'tcx>, depth_lvl: usize) {
        print_indented!(self, "AdtDef {", depth_lvl);
        print_indented!(self, format!("did: {:?}", adt_def.did()), depth_lvl + 1);
        print_indented!(self, format!("variants: {:?}", adt_def.variants()), depth_lvl + 1);
        print_indented!(self, format!("flags: {:?}", adt_def.flags()), depth_lvl + 1);
        print_indented!(self, format!("repr: {:?}", adt_def.repr()), depth_lvl + 1);
    }

    fn print_fru_info(&mut self, fru_info: &FruInfo<'tcx>, depth_lvl: usize) {
        print_indented!(self, "FruInfo {", depth_lvl);
        print_indented!(self, "base: ", depth_lvl + 1);
        self.print_expr(fru_info.base, depth_lvl + 2);
        print_indented!(self, "field_types: [", depth_lvl + 1);
        for ty in fru_info.field_types.iter() {
            print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 2);
        }
        print_indented!(self, "}", depth_lvl);
    }

    fn print_arm(&mut self, arm_id: ArmId, depth_lvl: usize) {
        print_indented!(self, "Arm {", depth_lvl);

        let arm = &self.thir.arms[arm_id];
        let Arm { pattern, guard, body, lint_level, scope, span } = arm;

        print_indented!(self, "pattern: ", depth_lvl + 1);
        self.print_pat(pattern, depth_lvl + 2);

        if let Some(guard) = *guard {
            print_indented!(self, "guard: ", depth_lvl + 1);
            self.print_expr(guard, depth_lvl + 2);
        } else {
            print_indented!(self, "guard: None", depth_lvl + 1);
        }

        print_indented!(self, "body: ", depth_lvl + 1);
        self.print_expr(*body, depth_lvl + 2);
        print_indented!(self, format!("lint_level: {:?}", lint_level), depth_lvl + 1);
        print_indented!(self, format!("scope: {:?}", scope), depth_lvl + 1);
        print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
        print_indented!(self, "}", depth_lvl);
    }

    fn print_pat(&mut self, pat: &Box<Pat<'tcx>>, depth_lvl: usize) {
        let Pat { ty, span, kind } = &**pat;

        print_indented!(self, "Pat: {", depth_lvl);
        print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 1);
        print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
        self.print_pat_kind(kind, depth_lvl + 1);
        print_indented!(self, "}", depth_lvl);
    }

    fn print_pat_kind(&mut self, pat_kind: &PatKind<'tcx>, depth_lvl: usize) {
        print_indented!(self, "kind: PatKind {", depth_lvl);

        match pat_kind {
            PatKind::Wild => {
                print_indented!(self, "Wild", depth_lvl + 1);
            }
            PatKind::Never => {
                print_indented!(self, "Never", depth_lvl + 1);
            }
            PatKind::AscribeUserType { ascription, subpattern } => {
                print_indented!(self, "AscribeUserType: {", depth_lvl + 1);
                print_indented!(self, format!("ascription: {:?}", ascription), depth_lvl + 2);
                print_indented!(self, "subpattern: ", depth_lvl + 2);
                self.print_pat(subpattern, depth_lvl + 3);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Binding { name, mode, var, ty, subpattern, is_primary } => {
                print_indented!(self, "Binding {", depth_lvl + 1);
                print_indented!(self, format!("name: {:?}", name), depth_lvl + 2);
                print_indented!(self, format!("mode: {:?}", mode), depth_lvl + 2);
                print_indented!(self, format!("var: {:?}", var), depth_lvl + 2);
                print_indented!(self, format!("ty: {:?}", ty), depth_lvl + 2);
                print_indented!(self, format!("is_primary: {:?}", is_primary), depth_lvl + 2);

                if let Some(subpattern) = subpattern {
                    print_indented!(self, "subpattern: Some( ", depth_lvl + 2);
                    self.print_pat(subpattern, depth_lvl + 3);
                    print_indented!(self, ")", depth_lvl + 2);
                } else {
                    print_indented!(self, "subpattern: None", depth_lvl + 2);
                }

                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Variant { adt_def, args, variant_index, subpatterns } => {
                print_indented!(self, "Variant {", depth_lvl + 1);
                print_indented!(self, "adt_def: ", depth_lvl + 2);
                self.print_adt_def(*adt_def, depth_lvl + 3);
                print_indented!(self, format!("args: {:?}", args), depth_lvl + 2);
                print_indented!(self, format!("variant_index: {:?}", variant_index), depth_lvl + 2);

                if subpatterns.len() > 0 {
                    print_indented!(self, "subpatterns: [", depth_lvl + 2);
                    for field_pat in subpatterns.iter() {
                        self.print_pat(&field_pat.pattern, depth_lvl + 3);
                    }
                    print_indented!(self, "]", depth_lvl + 2);
                } else {
                    print_indented!(self, "subpatterns: []", depth_lvl + 2);
                }

                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Leaf { subpatterns } => {
                print_indented!(self, "Leaf { ", depth_lvl + 1);
                print_indented!(self, "subpatterns: [", depth_lvl + 2);
                for field_pat in subpatterns.iter() {
                    self.print_pat(&field_pat.pattern, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Deref { subpattern } => {
                print_indented!(self, "Deref { ", depth_lvl + 1);
                print_indented!(self, "subpattern:", depth_lvl + 2);
                self.print_pat(subpattern, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::DerefPattern { subpattern, .. } => {
                print_indented!(self, "DerefPattern { ", depth_lvl + 1);
                print_indented!(self, "subpattern:", depth_lvl + 2);
                self.print_pat(subpattern, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Constant { value } => {
                print_indented!(self, "Constant {", depth_lvl + 1);
                print_indented!(self, format!("value: {:?}", value), depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::ExpandedConstant { def_id, is_inline, subpattern } => {
                print_indented!(self, "ExpandedConstant {", depth_lvl + 1);
                print_indented!(self, format!("def_id: {def_id:?}"), depth_lvl + 2);
                print_indented!(self, format!("is_inline: {is_inline:?}"), depth_lvl + 2);
                print_indented!(self, "subpattern:", depth_lvl + 2);
                self.print_pat(subpattern, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Range(pat_range) => {
                print_indented!(self, format!("Range ( {:?} )", pat_range), depth_lvl + 1);
            }
            PatKind::Slice { prefix, slice, suffix } => {
                print_indented!(self, "Slice {", depth_lvl + 1);

                print_indented!(self, "prefix: [", depth_lvl + 2);
                for prefix_pat in prefix.iter() {
                    self.print_pat(prefix_pat, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);

                if let Some(slice) = slice {
                    print_indented!(self, "slice: ", depth_lvl + 2);
                    self.print_pat(slice, depth_lvl + 3);
                }

                print_indented!(self, "suffix: [", depth_lvl + 2);
                for suffix_pat in suffix.iter() {
                    self.print_pat(suffix_pat, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);

                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Array { prefix, slice, suffix } => {
                print_indented!(self, "Array {", depth_lvl + 1);

                print_indented!(self, "prefix: [", depth_lvl + 2);
                for prefix_pat in prefix.iter() {
                    self.print_pat(prefix_pat, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);

                if let Some(slice) = slice {
                    print_indented!(self, "slice: ", depth_lvl + 2);
                    self.print_pat(slice, depth_lvl + 3);
                }

                print_indented!(self, "suffix: [", depth_lvl + 2);
                for suffix_pat in suffix.iter() {
                    self.print_pat(suffix_pat, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);

                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Or { pats } => {
                print_indented!(self, "Or {", depth_lvl + 1);
                print_indented!(self, "pats: [", depth_lvl + 2);
                for pat in pats.iter() {
                    self.print_pat(pat, depth_lvl + 3);
                }
                print_indented!(self, "]", depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            PatKind::Error(_) => {
                print_indented!(self, "Error", depth_lvl + 1);
            }
        }

        print_indented!(self, "}", depth_lvl);
    }

    fn print_closure_expr(&mut self, expr: &ClosureExpr<'tcx>, depth_lvl: usize) {
        let ClosureExpr { closure_id, args, upvars, movability, fake_reads } = expr;

        print_indented!(self, "ClosureExpr {", depth_lvl);
        print_indented!(self, format!("closure_id: {:?}", closure_id), depth_lvl + 1);
        print_indented!(self, format!("args: {:?}", args), depth_lvl + 1);

        if upvars.len() > 0 {
            print_indented!(self, "upvars: [", depth_lvl + 1);
            for upvar in upvars.iter() {
                self.print_expr(*upvar, depth_lvl + 2);
                print_indented!(self, ",", depth_lvl + 1);
            }
            print_indented!(self, "]", depth_lvl + 1);
        } else {
            print_indented!(self, "upvars: []", depth_lvl + 1);
        }

        print_indented!(self, format!("movability: {:?}", movability), depth_lvl + 1);

        if fake_reads.len() > 0 {
            print_indented!(self, "fake_reads: [", depth_lvl + 1);
            for (fake_read_expr, cause, hir_id) in fake_reads.iter() {
                print_indented!(self, "(", depth_lvl + 2);
                self.print_expr(*fake_read_expr, depth_lvl + 3);
                print_indented!(self, ",", depth_lvl + 2);
                print_indented!(self, format!("cause: {:?}", cause), depth_lvl + 3);
                print_indented!(self, ",", depth_lvl + 2);
                print_indented!(self, format!("hir_id: {:?}", hir_id), depth_lvl + 3);
                print_indented!(self, "),", depth_lvl + 2);
            }
            print_indented!(self, "]", depth_lvl + 1);
        } else {
            print_indented!(self, "fake_reads: []", depth_lvl + 1);
        }

        print_indented!(self, "}", depth_lvl);
    }

    fn print_inline_asm_expr(&mut self, expr: &InlineAsmExpr<'tcx>, depth_lvl: usize) {
        let InlineAsmExpr { asm_macro, template, operands, options, line_spans } = expr;

        print_indented!(self, "InlineAsmExpr {", depth_lvl);

        print_indented!(self, format!("asm_macro: {:?}", asm_macro), depth_lvl + 1);

        print_indented!(self, "template: [", depth_lvl + 1);
        for template_piece in template.iter() {
            print_indented!(self, format!("{:?}", template_piece), depth_lvl + 2);
        }
        print_indented!(self, "]", depth_lvl + 1);

        print_indented!(self, "operands: [", depth_lvl + 1);
        for operand in operands.iter() {
            self.print_inline_operand(operand, depth_lvl + 2);
        }
        print_indented!(self, "]", depth_lvl + 1);

        print_indented!(self, format!("options: {:?}", options), depth_lvl + 1);
        print_indented!(self, format!("line_spans: {:?}", line_spans), depth_lvl + 1);
    }

    fn print_inline_operand(&mut self, operand: &InlineAsmOperand<'tcx>, depth_lvl: usize) {
        match operand {
            InlineAsmOperand::In { reg, expr } => {
                print_indented!(self, "InlineAsmOperand::In {", depth_lvl);
                print_indented!(self, format!("reg: {:?}", reg), depth_lvl + 1);
                print_indented!(self, "expr: ", depth_lvl + 1);
                self.print_expr(*expr, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::Out { reg, late, expr } => {
                print_indented!(self, "InlineAsmOperand::Out {", depth_lvl);
                print_indented!(self, format!("reg: {:?}", reg), depth_lvl + 1);
                print_indented!(self, format!("late: {:?}", late), depth_lvl + 1);

                if let Some(out) = expr {
                    print_indented!(self, "place: Some( ", depth_lvl + 1);
                    self.print_expr(*out, depth_lvl + 2);
                    print_indented!(self, ")", depth_lvl + 1);
                } else {
                    print_indented!(self, "place: None", depth_lvl + 1);
                }
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::InOut { reg, late, expr } => {
                print_indented!(self, "InlineAsmOperand::InOut {", depth_lvl);
                print_indented!(self, format!("reg: {:?}", reg), depth_lvl + 1);
                print_indented!(self, format!("late: {:?}", late), depth_lvl + 1);
                print_indented!(self, "expr: ", depth_lvl + 1);
                self.print_expr(*expr, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::SplitInOut { reg, late, in_expr, out_expr } => {
                print_indented!(self, "InlineAsmOperand::SplitInOut {", depth_lvl);
                print_indented!(self, format!("reg: {:?}", reg), depth_lvl + 1);
                print_indented!(self, format!("late: {:?}", late), depth_lvl + 1);
                print_indented!(self, "in_expr: ", depth_lvl + 1);
                self.print_expr(*in_expr, depth_lvl + 2);

                if let Some(out_expr) = out_expr {
                    print_indented!(self, "out_expr: Some( ", depth_lvl + 1);
                    self.print_expr(*out_expr, depth_lvl + 2);
                    print_indented!(self, ")", depth_lvl + 1);
                } else {
                    print_indented!(self, "out_expr: None", depth_lvl + 1);
                }

                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::Const { value, span } => {
                print_indented!(self, "InlineAsmOperand::Const {", depth_lvl);
                print_indented!(self, format!("value: {:?}", value), depth_lvl + 1);
                print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::SymFn { value, span } => {
                print_indented!(self, "InlineAsmOperand::SymFn {", depth_lvl);
                print_indented!(self, format!("value: {:?}", *value), depth_lvl + 1);
                print_indented!(self, format!("span: {:?}", span), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::SymStatic { def_id } => {
                print_indented!(self, "InlineAsmOperand::SymStatic {", depth_lvl);
                print_indented!(self, format!("def_id: {:?}", def_id), depth_lvl + 1);
                print_indented!(self, "}", depth_lvl + 1);
            }
            InlineAsmOperand::Label { block } => {
                print_indented!(self, "InlineAsmOperand::Block {", depth_lvl);
                print_indented!(self, "block:", depth_lvl + 1);
                self.print_block(*block, depth_lvl + 2);
                print_indented!(self, "}", depth_lvl + 1);
            }
        }
    }
}
