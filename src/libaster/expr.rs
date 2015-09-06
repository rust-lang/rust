// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::IntoIterator;

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span, Spanned, respan};
use syntax::ptr::P;

use block::BlockBuilder;
use ident::ToIdent;
use invoke::{Invoke, Identity};
use lit::LitBuilder;
use path::{IntoPath, PathBuilder};
use qpath::QPathBuilder;
use str::ToInternedString;
use ty::TyBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct ExprBuilder<F=Identity> {
    callback: F,
    span: Span,
}

impl ExprBuilder {
    pub fn new() -> Self {
        ExprBuilder::new_with_callback(Identity)
    }
}

impl<F> ExprBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    pub fn new_with_callback(callback: F) -> Self {
        ExprBuilder {
            callback: callback,
            span: DUMMY_SP,
        }
    }

    pub fn build(self, expr: P<ast::Expr>) -> F::Result {
        self.callback.invoke(expr)
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn build_expr_(self, expr: ast::Expr_) -> F::Result {
        let expr = P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: expr,
            span: self.span,
        });
        self.build(expr)
    }

    pub fn build_path(self, path: ast::Path) -> F::Result {
        self.build_expr_(ast::Expr_::ExprPath(None, path))
    }

    pub fn build_qpath(self, qself: ast::QSelf, path: ast::Path) -> F::Result {
        self.build_expr_(ast::Expr_::ExprPath(Some(qself), path))
    }

    pub fn path(self) -> PathBuilder<Self> {
        PathBuilder::new_with_callback(self)
    }

    pub fn qpath(self) -> QPathBuilder<Self> {
        QPathBuilder::new_with_callback(self)
    }

    pub fn id<I>(self, id: I) -> F::Result
        where I: ToIdent
    {
        self.path().id(id).build()
    }

    pub fn build_lit(self, lit: P<ast::Lit>) -> F::Result {
        self.build_expr_(ast::Expr_::ExprLit(lit))
    }

    pub fn lit(self) -> LitBuilder<Self> {
        LitBuilder::new_with_callback(self)
    }

    pub fn bool(self, value: bool) -> F::Result {
        self.lit().bool(value)
    }

    pub fn int(self, value: i64) -> F::Result {
        self.lit().int(value)
    }

    pub fn isize(self, value: isize) -> F::Result {
        self.lit().isize(value)
    }

    pub fn i8(self, value: i8) -> F::Result {
        self.lit().i8(value)
    }

    pub fn i16(self, value: i16) -> F::Result {
        self.lit().i16(value)
    }

    pub fn i32(self, value: i32) -> F::Result {
        self.lit().i32(value)
    }

    pub fn i64(self, value: i64) -> F::Result {
        self.lit().i64(value)
    }

    pub fn usize(self, value: usize) -> F::Result {
        self.lit().usize(value)
    }

    pub fn u8(self, value: u8) -> F::Result {
        self.lit().u8(value)
    }

    pub fn u16(self, value: u16) -> F::Result {
        self.lit().u16(value)
    }

    pub fn u32(self, value: u32) -> F::Result {
        self.lit().u32(value)
    }

    pub fn u64(self, value: u64) -> F::Result {
        self.lit().u64(value)
    }

    pub fn f32<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        self.lit().f32(value)
    }

    pub fn f64<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        self.lit().f64(value)
    }

    pub fn str<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        self.lit().str(value)
    }

    pub fn build_unary(self, unop: ast::UnOp, expr: P<ast::Expr>) -> F::Result {
        self.build_expr_(ast::ExprUnary(unop, expr))
    }

    pub fn build_box(self, expr: P<ast::Expr>) -> F::Result {
        self.build_unary(ast::UnUniq, expr)
    }

    pub fn build_deref(self, expr: P<ast::Expr>) -> F::Result {
        self.build_unary(ast::UnDeref, expr)
    }

    pub fn build_not(self, expr: P<ast::Expr>) -> F::Result {
        self.build_unary(ast::UnNot, expr)
    }

    pub fn build_neg(self, expr: P<ast::Expr>) -> F::Result {
        self.build_unary(ast::UnNeg, expr)
    }

    pub fn unary(self, unop: ast::UnOp) -> ExprBuilder<ExprUnaryBuilder<F>> {
        ExprBuilder::new_with_callback(ExprUnaryBuilder {
            builder: self,
            unop: unop,
        })
    }

    // FIXME: Disabled for now until the `box` keyword is stablized.
    /*
    pub fn box_(self) -> ExprBuilder<ExprUnaryBuilder<F>> {
        self.unary(ast::UnUniq)
    }
    */

    pub fn deref(self) -> ExprBuilder<ExprUnaryBuilder<F>> {
        self.unary(ast::UnDeref)
    }

    pub fn not(self) -> ExprBuilder<ExprUnaryBuilder<F>> {
        self.unary(ast::UnNot)
    }

    pub fn neg(self) -> ExprBuilder<ExprUnaryBuilder<F>> {
        self.unary(ast::UnNeg)
    }

    pub fn build_binary(
        self,
        binop: ast::BinOp_,
        lhs: P<ast::Expr>,
        rhs: P<ast::Expr>,
    ) -> F::Result {
        let binop = respan(self.span, binop);
        self.build_expr_(ast::Expr_::ExprBinary(binop, lhs, rhs))
    }

    pub fn build_add(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiAdd, lhs, rhs)
    }

    pub fn build_sub(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiSub, lhs, rhs)
    }

    pub fn build_mul(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiMul, lhs, rhs)
    }

    pub fn build_div(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiDiv, lhs, rhs)
    }

    pub fn build_rem(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiRem, lhs, rhs)
    }

    pub fn build_and(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiAnd, lhs, rhs)
    }

    pub fn build_or(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiOr, lhs, rhs)
    }

    pub fn build_bit_xor(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiBitXor, lhs, rhs)
    }

    pub fn build_bit_and(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiBitAnd, lhs, rhs)
    }

    pub fn build_bit_or(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiBitOr, lhs, rhs)
    }

    pub fn build_shl(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiShl, lhs, rhs)
    }

    pub fn build_shr(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiShr, lhs, rhs)
    }

    pub fn build_eq(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiEq, lhs, rhs)
    }

    pub fn build_lt(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiLt, lhs, rhs)
    }

    pub fn build_le(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiLe, lhs, rhs)
    }

    pub fn build_ne(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiNe, lhs, rhs)
    }

    pub fn build_ge(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiGe, lhs, rhs)
    }

    pub fn build_gt(self, lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> F::Result {
        self.build_binary(ast::BinOp_::BiGt, lhs, rhs)
    }

    pub fn binary(self, binop: ast::BinOp_) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        ExprBuilder::new_with_callback(ExprBinaryLhsBuilder {
            builder: self,
            binop: binop,
        })
    }

    pub fn add(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiAdd)
    }

    pub fn sub(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiSub)
    }

    pub fn mul(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiMul)
    }

    pub fn div(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiDiv)
    }

    pub fn rem(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiRem)
    }

    pub fn and(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiAnd)
    }

    pub fn or(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiOr)
    }

    pub fn bit_xor(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiBitXor)
    }

    pub fn bit_and(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiBitAnd)
    }

    pub fn bit_or(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiBitOr)
    }

    pub fn shl(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiShl)
    }

    pub fn shr(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiShr)
    }

    pub fn eq(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiEq)
    }

    pub fn lt(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiLt)
    }

    pub fn le(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiLe)
    }

    pub fn ne(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiNe)
    }

    pub fn ge(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiGe)
    }

    pub fn gt(self) -> ExprBuilder<ExprBinaryLhsBuilder<F>> {
        self.binary(ast::BinOp_::BiGt)
    }

    pub fn addr_of(self) -> ExprBuilder<ExprAddrOfBuilder<F>> {
        ExprBuilder::new_with_callback(ExprAddrOfBuilder {
            builder: self,
            mutability: ast::Mutability::MutImmutable,
        })
    }

    pub fn mut_addr_of(self) -> ExprBuilder<ExprAddrOfBuilder<F>> {
        ExprBuilder::new_with_callback(ExprAddrOfBuilder {
            builder: self,
            mutability: ast::Mutability::MutMutable,
        })
    }

    pub fn unit(self) -> F::Result {
        self.tuple().build()
    }

    pub fn tuple(self) -> ExprTupleBuilder<F> {
        ExprTupleBuilder {
            builder: self,
            exprs: Vec::new(),
        }
    }

    pub fn struct_path<P>(self, path: P) -> ExprStructPathBuilder<F>
        where P: IntoPath,
    {
        let span = self.span;
        let path = path.into_path();
        ExprStructPathBuilder {
            builder: self,
            span: span,
            path: path,
            fields: vec![],
        }
    }

    pub fn struct_(self) -> PathBuilder<ExprStructBuilder<F>> {
        PathBuilder::new_with_callback(ExprStructBuilder {
            builder: self,
        })
    }

    pub fn self_(self) -> F::Result {
        self.id("self")
    }

    pub fn none(self) -> F::Result {
        self.path()
            .global()
            .id("std").id("option").id("Option").id("None")
            .build()
    }

    pub fn some(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("option").id("Option").id("Some")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn ok(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("result").id("Result").id("Ok")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn err(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("result").id("Result").id("Err")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn phantom_data(self) -> F::Result {
        self.path()
            .global()
            .ids(&["std", "marker", "PhantomData"])
            .build()
    }

    pub fn call(self) -> ExprBuilder<ExprCallBuilder<F>> {
        ExprBuilder::new_with_callback(ExprCallBuilder {
            builder: self,
        })
    }

    pub fn method_call<I>(self, id: I) -> ExprBuilder<ExprMethodCallBuilder<F>>
        where I: ToIdent,
    {
        let id = respan(self.span, id.to_ident());
        ExprBuilder::new_with_callback(ExprMethodCallBuilder {
            builder: self,
            id: id,
        })
    }

    pub fn block(self) -> BlockBuilder<Self> {
        BlockBuilder::new_with_callback(self)
    }

    pub fn paren(self) -> ExprBuilder<ExprParenBuilder<F>> {
        ExprBuilder::new_with_callback(ExprParenBuilder {
            builder: self,
        })
    }

    pub fn field<I>(self, id: I) -> ExprBuilder<ExprFieldBuilder<F>>
        where I: ToIdent,
    {
        let id = respan(self.span, id.to_ident());
        ExprBuilder::new_with_callback(ExprFieldBuilder {
            builder: self,
            id: id,
        })
    }

    pub fn tup_field(self, index: usize) -> ExprBuilder<ExprTupFieldBuilder<F>> {
        let index = respan(self.span, index);
        ExprBuilder::new_with_callback(ExprTupFieldBuilder {
            builder: self,
            index: index,
        })
    }

    pub fn box_(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("boxed").id("Box").id("new")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn rc(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("rc").id("Rc").id("new")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn arc(self) -> ExprBuilder<ExprPathBuilder<F>> {
        let path = PathBuilder::new()
            .global()
            .id("std").id("arc").id("Arc").id("new")
            .build();

        ExprBuilder::new_with_callback(ExprPathBuilder {
            builder: self,
            path: path,
        })
    }

    pub fn slice(self) -> ExprSliceBuilder<F> {
        ExprSliceBuilder {
            builder: self,
            exprs: Vec::new(),
        }
    }

    pub fn vec(self) -> ExprSliceBuilder<ExprVecBuilder<F>> {
        ExprBuilder::new_with_callback(ExprVecBuilder {
            builder: self,
        }).slice()
    }
}

impl<F> Invoke<P<ast::Lit>> for ExprBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, lit: P<ast::Lit>) -> F::Result {
        self.build_lit(lit)
    }
}

impl<F> Invoke<ast::Path> for ExprBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, path: ast::Path) -> F::Result {
        self.build_path(path)
    }
}

impl<F> Invoke<(ast::QSelf, ast::Path)> for ExprBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, (qself, path): (ast::QSelf, ast::Path)) -> F::Result {
        self.build_qpath(qself, path)
    }
}

impl<F> Invoke<P<ast::Block>> for ExprBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, block: P<ast::Block>) -> F::Result {
        self.build_expr_(ast::ExprBlock(block))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprUnaryBuilder<F> {
    builder: ExprBuilder<F>,
    unop: ast::UnOp,
}

impl<F> Invoke<P<ast::Expr>> for ExprUnaryBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.builder.build_unary(self.unop, expr)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprBinaryLhsBuilder<F> {
    builder: ExprBuilder<F>,
    binop: ast::BinOp_,
}

impl<F> Invoke<P<ast::Expr>> for ExprBinaryLhsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = ExprBuilder<ExprBinaryRhsBuilder<F>>;

    fn invoke(self, lhs: P<ast::Expr>) -> ExprBuilder<ExprBinaryRhsBuilder<F>> {
        ExprBuilder::new_with_callback(ExprBinaryRhsBuilder {
            builder: self.builder,
            binop: self.binop,
            lhs: lhs,
        })
    }
}

pub struct ExprBinaryRhsBuilder<F> {
    builder: ExprBuilder<F>,
    binop: ast::BinOp_,
    lhs: P<ast::Expr>,
}

impl<F> Invoke<P<ast::Expr>> for ExprBinaryRhsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, rhs: P<ast::Expr>) -> F::Result {
        self.builder.build_binary(self.binop, self.lhs, rhs)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprTupleBuilder<F> {
    builder: ExprBuilder<F>,
    exprs: Vec<P<ast::Expr>>,
}

impl<F: Invoke<P<ast::Expr>>> ExprTupleBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    pub fn with_exprs<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Expr>>,
    {
        self.exprs.extend(iter);
        self
    }

    pub fn expr(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.builder.build_expr_(ast::ExprTup(self.exprs))
    }
}

impl<F> Invoke<P<ast::Expr>> for ExprTupleBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    type Result = ExprTupleBuilder<F>;

    fn invoke(mut self, expr: P<ast::Expr>) -> Self {
        self.exprs.push(expr);
        self
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprStructBuilder<F> {
    builder: ExprBuilder<F>,
}

impl<F> Invoke<ast::Path> for ExprStructBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    type Result = ExprStructPathBuilder<F>;

    fn invoke(self, path: ast::Path) -> ExprStructPathBuilder<F> {
        self.builder.struct_path(path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprStructPathBuilder<F> {
    builder: ExprBuilder<F>,
    span: Span,
    path: ast::Path,
    fields: Vec<ast::Field>,
}

impl<F> ExprStructPathBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_fields<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=ast::Field>,
    {
        self.fields.extend(iter);
        self
    }

    pub fn with_id_exprs<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=(ast::Ident, P<ast::Expr>)>,
    {
        for (id, expr) in iter {
            self = self.field(id).build(expr);
        }

        self
    }

    pub fn field<I>(self, id: I) -> ExprBuilder<ExprStructFieldBuilder<I, F>>
        where I: ToIdent,
    {
        ExprBuilder::new_with_callback(ExprStructFieldBuilder {
            builder: self,
            id: id,
        })
    }

    pub fn build_with(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        let expr_ = ast::ExprStruct(self.path, self.fields, None);
        self.builder.build_expr_(expr_)
    }
}

impl<F> Invoke<P<ast::Expr>> for ExprStructPathBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        let expr_ = ast::ExprStruct(self.path, self.fields, Some(expr));
        self.builder.build_expr_(expr_)
    }
}

pub struct ExprStructFieldBuilder<I, F> {
    builder: ExprStructPathBuilder<F>,
    id: I,
}

impl<I, F> Invoke<P<ast::Expr>> for ExprStructFieldBuilder<I, F>
    where I: ToIdent,
          F: Invoke<P<ast::Expr>>,
{
    type Result = ExprStructPathBuilder<F>;

    fn invoke(mut self, expr: P<ast::Expr>) -> ExprStructPathBuilder<F> {
        let field = ast::Field {
            ident: respan(self.builder.span, self.id.to_ident()),
            expr: expr,
            span: self.builder.span,
        };
        self.builder.fields.push(field);
        self.builder
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprCallBuilder<F> {
    builder: ExprBuilder<F>,
}

impl<F> Invoke<P<ast::Expr>> for ExprCallBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = ExprCallArgsBuilder<F>;

    fn invoke(self, expr: P<ast::Expr>) -> ExprCallArgsBuilder<F> {
        ExprCallArgsBuilder {
            builder: self.builder,
            fn_: expr,
            args: vec![],
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprCallArgsBuilder<F> {
    builder: ExprBuilder<F>,
    fn_: P<ast::Expr>,
    args: Vec<P<ast::Expr>>,
}

impl<F> ExprCallArgsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    pub fn with_args<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Expr>>,
    {
        self.args.extend(iter);
        self
    }

    pub fn with_arg(mut self, arg: P<ast::Expr>) -> Self {
        self.args.push(arg);
        self
    }

    pub fn arg(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.builder.build_expr_(ast::ExprCall(self.fn_, self.args))
    }
}

impl<F> Invoke<P<ast::Expr>> for ExprCallArgsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = Self;

    fn invoke(self, arg: P<ast::Expr>) -> Self {
        self.with_arg(arg)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprMethodCallBuilder<F> {
    builder: ExprBuilder<F>,
    id: ast::SpannedIdent,
}

impl<F> Invoke<P<ast::Expr>> for ExprMethodCallBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = ExprMethodCallArgsBuilder<F>;

    fn invoke(self, expr: P<ast::Expr>) -> ExprMethodCallArgsBuilder<F> {
        ExprMethodCallArgsBuilder {
            builder: self.builder,
            id: self.id,
            tys: vec![],
            args: vec![expr],
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprMethodCallArgsBuilder<F> {
    builder: ExprBuilder<F>,
    id: ast::SpannedIdent,
    tys: Vec<P<ast::Ty>>,
    args: Vec<P<ast::Expr>>,
}

impl<F> ExprMethodCallArgsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    pub fn with_tys<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Ty>>,
    {
        self.tys.extend(iter);
        self
    }

    pub fn with_ty(mut self, ty: P<ast::Ty>) -> Self {
        self.tys.push(ty);
        self
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn with_args<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Expr>>,
    {
        self.args.extend(iter);
        self
    }

    pub fn with_arg(mut self, arg: P<ast::Expr>) -> Self {
        self.args.push(arg);
        self
    }

    pub fn arg(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.builder.build_expr_(ast::ExprMethodCall(self.id, self.tys, self.args))
    }
}

impl<F> Invoke<P<ast::Ty>> for ExprMethodCallArgsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = Self;

    fn invoke(self, ty: P<ast::Ty>) -> Self {
        self.with_ty(ty)
    }
}

impl<F> Invoke<P<ast::Expr>> for ExprMethodCallArgsBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = Self;

    fn invoke(self, arg: P<ast::Expr>) -> Self {
        self.with_arg(arg)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprAddrOfBuilder<F> {
    builder: ExprBuilder<F>,
    mutability: ast::Mutability,
}

impl<F> Invoke<P<ast::Expr>> for ExprAddrOfBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.builder.build_expr_(ast::ExprAddrOf(self.mutability, expr))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprPathBuilder<F> {
    builder: ExprBuilder<F>,
    path: ast::Path,
}

impl<F> Invoke<P<ast::Expr>> for ExprPathBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, arg: P<ast::Expr>) -> F::Result {
        self.builder.call()
            .build_path(self.path)
            .with_arg(arg)
            .build()
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprParenBuilder<F> {
    builder: ExprBuilder<F>,
}

impl<F> Invoke<P<ast::Expr>> for ExprParenBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.builder.build_expr_(ast::ExprParen(expr))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprFieldBuilder<F> {
    builder: ExprBuilder<F>,
    id: ast::SpannedIdent,
}

impl<F> Invoke<P<ast::Expr>> for ExprFieldBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.builder.build_expr_(ast::ExprField(expr, self.id))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprTupFieldBuilder<F> {
    builder: ExprBuilder<F>,
    index: Spanned<usize>,
}

impl<F> Invoke<P<ast::Expr>> for ExprTupFieldBuilder<F>
    where F: Invoke<P<ast::Expr>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.builder.build_expr_(ast::ExprTupField(expr, self.index))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprSliceBuilder<F> {
    builder: ExprBuilder<F>,
    exprs: Vec<P<ast::Expr>>,
}

impl<F: Invoke<P<ast::Expr>>> ExprSliceBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    pub fn with_exprs<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Expr>>,
    {
        self.exprs.extend(iter);
        self
    }

    pub fn expr(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.builder.build_expr_(ast::ExprVec(self.exprs))
    }
}

impl<F> Invoke<P<ast::Expr>> for ExprSliceBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    type Result = ExprSliceBuilder<F>;

    fn invoke(mut self, expr: P<ast::Expr>) -> Self {
        self.exprs.push(expr);
        self
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ExprVecBuilder<F> {
    builder: ExprBuilder<F>,
}

impl<F> Invoke<P<ast::Expr>> for ExprVecBuilder<F>
    where F: Invoke<P<ast::Expr>>
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        let qpath = ExprBuilder::new().qpath()
            .ty().slice().infer()
            .id("into_vec");

        self.builder.call()
            .build(qpath)
            .arg().box_().build(expr)
            .build()
    }
}
