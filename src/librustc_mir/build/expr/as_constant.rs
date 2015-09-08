// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See docs in build/expr/mod.rs

use rustc_data_structures::fnv::FnvHashMap;

use build::{Builder};
use hair::*;
use repr::*;

impl<H:Hair> Builder<H> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    pub fn as_constant<M>(&mut self, expr: M) -> Constant<H>
        where M: Mirror<H, Output=Expr<H>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_constant(expr)
    }

    fn expr_as_constant(&mut self, expr: Expr<H>) -> Constant<H> {
        let this = self;
        let Expr { ty: _, temp_lifetime: _, span, kind } = expr;
        let kind = match kind {
            ExprKind::Scope { extent: _, value } => {
                return this.as_constant(value);
            }
            ExprKind::Paren { arg } => {
                return this.as_constant(arg);
            }
            ExprKind::Literal { literal } => {
                ConstantKind::Literal(literal)
            }
            ExprKind::Vec { fields } => {
                let fields = this.as_constants(fields);
                ConstantKind::Aggregate(AggregateKind::Vec, fields)
            }
            ExprKind::Tuple { fields } => {
                let fields = this.as_constants(fields);
                ConstantKind::Aggregate(AggregateKind::Tuple, fields)
            }
            ExprKind::Adt { adt_def, variant_index, substs, fields, base: None } => {
                let field_names = this.hir.fields(adt_def, variant_index);
                let fields = this.named_field_constants(field_names, fields);
                ConstantKind::Aggregate(AggregateKind::Adt(adt_def, variant_index, substs), fields)
            }
            ExprKind::Repeat { value, count } => {
                let value = Box::new(this.as_constant(value));
                let count = Box::new(this.as_constant(count));
                ConstantKind::Repeat(value, count)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs = Box::new(this.as_constant(lhs));
                let rhs = Box::new(this.as_constant(rhs));
                ConstantKind::BinaryOp(op, lhs, rhs)
            }
            ExprKind::Unary { op, arg } => {
                let arg = Box::new(this.as_constant(arg));
                ConstantKind::UnaryOp(op, arg)
            }
            ExprKind::Field { lhs, name } => {
                let lhs = this.as_constant(lhs);
                ConstantKind::Projection(
                    Box::new(ConstantProjection {
                        base: lhs,
                        elem: ProjectionElem::Field(name),
                    }))
            }
            ExprKind::Deref { arg } => {
                let arg = this.as_constant(arg);
                ConstantKind::Projection(
                    Box::new(ConstantProjection {
                        base: arg,
                        elem: ProjectionElem::Deref,
                    }))
            }
            ExprKind::Call { fun, args } => {
                let fun = this.as_constant(fun);
                let args = this.as_constants(args);
                ConstantKind::Call(Box::new(fun), args)
            }
            _ => {
                this.hir.span_bug(
                    span,
                    &format!("expression is not a valid constant {:?}", kind));
            }
        };
        Constant { span: span, kind: kind }
    }

    fn as_constants(&mut self,
                    exprs: Vec<ExprRef<H>>)
                    -> Vec<Constant<H>>
    {
        exprs.into_iter().map(|expr| self.as_constant(expr)).collect()
    }

    fn named_field_constants(&mut self,
                             field_names: Vec<Field<H>>,
                             field_exprs: Vec<FieldExprRef<H>>)
                             -> Vec<Constant<H>>
    {
        let fields_map: FnvHashMap<_, _> =
            field_exprs.into_iter()
                       .map(|f| (f.name, self.as_constant(f.expr)))
                       .collect();

        let fields: Vec<_> =
            field_names.into_iter()
                       .map(|n| fields_map[&n].clone())
                       .collect();

        fields
    }
}
