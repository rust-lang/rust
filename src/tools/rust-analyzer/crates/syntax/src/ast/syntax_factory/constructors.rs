//! Wrappers over [`make`] constructors
use itertools::Itertools;

use crate::{
    ast::{self, make, HasGenericParams, HasName, HasTypeBounds, HasVisibility},
    syntax_editor::SyntaxMappingBuilder,
    AstNode, NodeOrToken, SyntaxKind, SyntaxNode, SyntaxToken,
};

use super::SyntaxFactory;

impl SyntaxFactory {
    pub fn name(&self, name: &str) -> ast::Name {
        make::name(name).clone_for_update()
    }

    pub fn ty(&self, text: &str) -> ast::Type {
        make::ty(text).clone_for_update()
    }

    pub fn ty_infer(&self) -> ast::InferType {
        let ast::Type::InferType(ast) = make::ty_placeholder().clone_for_update() else {
            unreachable!()
        };

        ast
    }

    pub fn type_param(
        &self,
        name: ast::Name,
        bounds: Option<ast::TypeBoundList>,
    ) -> ast::TypeParam {
        let ast = make::type_param(name.clone(), bounds.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            if let Some(input) = bounds {
                builder.map_node(
                    input.syntax().clone(),
                    ast.type_bound_list().unwrap().syntax().clone(),
                );
            }
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn ident_pat(&self, ref_: bool, mut_: bool, name: ast::Name) -> ast::IdentPat {
        let ast = make::ident_pat(ref_, mut_, name.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn block_expr(
        &self,
        stmts: impl IntoIterator<Item = ast::Stmt>,
        tail_expr: Option<ast::Expr>,
    ) -> ast::BlockExpr {
        let stmts = stmts.into_iter().collect_vec();
        let mut input = stmts.iter().map(|it| it.syntax().clone()).collect_vec();

        let ast = make::block_expr(stmts, tail_expr.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let stmt_list = ast.stmt_list().unwrap();
            let mut builder = SyntaxMappingBuilder::new(stmt_list.syntax().clone());

            if let Some(input) = tail_expr {
                builder.map_node(
                    input.syntax().clone(),
                    stmt_list.tail_expr().unwrap().syntax().clone(),
                );
            } else if let Some(ast_tail) = stmt_list.tail_expr() {
                // The parser interpreted the last statement (probably a statement with a block) as an Expr
                let last_stmt = input.pop().unwrap();

                builder.map_node(last_stmt, ast_tail.syntax().clone());
            }

            builder.map_children(
                input.into_iter(),
                stmt_list.statements().map(|it| it.syntax().clone()),
            );

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_empty_block(&self) -> ast::BlockExpr {
        ast::BlockExpr { syntax: make::expr_empty_block().syntax().clone_for_update() }
    }

    pub fn expr_bin(&self, lhs: ast::Expr, op: ast::BinaryOp, rhs: ast::Expr) -> ast::BinExpr {
        let ast::Expr::BinExpr(ast) =
            make::expr_bin_op(lhs.clone(), op, rhs.clone()).clone_for_update()
        else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(lhs.syntax().clone(), ast.lhs().unwrap().syntax().clone());
            builder.map_node(rhs.syntax().clone(), ast.rhs().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_path(&self, path: ast::Path) -> ast::Expr {
        let ast::Expr::PathExpr(ast) = make::expr_path(path.clone()).clone_for_update() else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(path.syntax().clone(), ast.path().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast.into()
    }

    pub fn expr_ref(&self, expr: ast::Expr, exclusive: bool) -> ast::Expr {
        let ast::Expr::RefExpr(ast) = make::expr_ref(expr.clone(), exclusive).clone_for_update()
        else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast.into()
    }

    pub fn expr_return(&self, expr: Option<ast::Expr>) -> ast::ReturnExpr {
        let ast::Expr::ReturnExpr(ast) = make::expr_return(expr.clone()).clone_for_update() else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(input) = expr {
                builder.map_node(input.syntax().clone(), ast.expr().unwrap().syntax().clone());
            }
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn let_stmt(
        &self,
        pattern: ast::Pat,
        ty: Option<ast::Type>,
        initializer: Option<ast::Expr>,
    ) -> ast::LetStmt {
        let ast =
            make::let_stmt(pattern.clone(), ty.clone(), initializer.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(pattern.syntax().clone(), ast.pat().unwrap().syntax().clone());
            if let Some(input) = ty {
                builder.map_node(input.syntax().clone(), ast.ty().unwrap().syntax().clone());
            }
            if let Some(input) = initializer {
                builder
                    .map_node(input.syntax().clone(), ast.initializer().unwrap().syntax().clone());
            }
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn item_const(
        &self,
        visibility: Option<ast::Visibility>,
        name: ast::Name,
        ty: ast::Type,
        expr: ast::Expr,
    ) -> ast::Const {
        let ast = make::item_const(visibility.clone(), name.clone(), ty.clone(), expr.clone())
            .clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }
            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            builder.map_node(ty.syntax().clone(), ast.ty().unwrap().syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.body().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn item_static(
        &self,
        visibility: Option<ast::Visibility>,
        is_unsafe: bool,
        is_mut: bool,
        name: ast::Name,
        ty: ast::Type,
        expr: Option<ast::Expr>,
    ) -> ast::Static {
        let ast = make::item_static(
            visibility.clone(),
            is_unsafe,
            is_mut,
            name.clone(),
            ty.clone(),
            expr.clone(),
        )
        .clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }

            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            builder.map_node(ty.syntax().clone(), ast.ty().unwrap().syntax().clone());

            if let Some(expr) = expr {
                builder.map_node(expr.syntax().clone(), ast.body().unwrap().syntax().clone());
            }
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn turbofish_generic_arg_list(
        &self,
        args: impl IntoIterator<Item = ast::GenericArg> + Clone,
    ) -> ast::GenericArgList {
        let ast = make::turbofish_generic_arg_list(args.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(
                args.into_iter().map(|arg| arg.syntax().clone()),
                ast.generic_args().map(|arg| arg.syntax().clone()),
            );
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn record_field_list(
        &self,
        fields: impl IntoIterator<Item = ast::RecordField>,
    ) -> ast::RecordFieldList {
        let fields: Vec<ast::RecordField> = fields.into_iter().collect();
        let input: Vec<_> = fields.iter().map(|it| it.syntax().clone()).collect();
        let ast = make::record_field_list(fields).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());

            builder.map_children(input.into_iter(), ast.fields().map(|it| it.syntax().clone()));

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn record_field(
        &self,
        visibility: Option<ast::Visibility>,
        name: ast::Name,
        ty: ast::Type,
    ) -> ast::RecordField {
        let ast =
            make::record_field(visibility.clone(), name.clone(), ty.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }

            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());
            builder.map_node(ty.syntax().clone(), ast.ty().unwrap().syntax().clone());

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn tuple_field_list(
        &self,
        fields: impl IntoIterator<Item = ast::TupleField>,
    ) -> ast::TupleFieldList {
        let fields: Vec<ast::TupleField> = fields.into_iter().collect();
        let input: Vec<_> = fields.iter().map(|it| it.syntax().clone()).collect();
        let ast = make::tuple_field_list(fields).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());

            builder.map_children(input.into_iter(), ast.fields().map(|it| it.syntax().clone()));

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn tuple_field(
        &self,
        visibility: Option<ast::Visibility>,
        ty: ast::Type,
    ) -> ast::TupleField {
        let ast = make::tuple_field(visibility.clone(), ty.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }

            builder.map_node(ty.syntax().clone(), ast.ty().unwrap().syntax().clone());

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn item_enum(
        &self,
        visibility: Option<ast::Visibility>,
        name: ast::Name,
        generic_param_list: Option<ast::GenericParamList>,
        where_clause: Option<ast::WhereClause>,
        variant_list: ast::VariantList,
    ) -> ast::Enum {
        let ast = make::enum_(
            visibility.clone(),
            name.clone(),
            generic_param_list.clone(),
            where_clause.clone(),
            variant_list.clone(),
        )
        .clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }

            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());

            if let Some(generic_param_list) = generic_param_list {
                builder.map_node(
                    generic_param_list.syntax().clone(),
                    ast.generic_param_list().unwrap().syntax().clone(),
                );
            }

            if let Some(where_clause) = where_clause {
                builder.map_node(
                    where_clause.syntax().clone(),
                    ast.where_clause().unwrap().syntax().clone(),
                );
            }

            builder.map_node(
                variant_list.syntax().clone(),
                ast.variant_list().unwrap().syntax().clone(),
            );

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn variant_list(
        &self,
        variants: impl IntoIterator<Item = ast::Variant>,
    ) -> ast::VariantList {
        let variants: Vec<ast::Variant> = variants.into_iter().collect();
        let input: Vec<_> = variants.iter().map(|it| it.syntax().clone()).collect();
        let ast = make::variant_list(variants).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());

            builder.map_children(input.into_iter(), ast.variants().map(|it| it.syntax().clone()));

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn variant(
        &self,
        visibility: Option<ast::Visibility>,
        name: ast::Name,
        field_list: Option<ast::FieldList>,
        discriminant: Option<ast::Expr>,
    ) -> ast::Variant {
        let ast = make::variant(
            visibility.clone(),
            name.clone(),
            field_list.clone(),
            discriminant.clone(),
        )
        .clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            if let Some(visibility) = visibility {
                builder.map_node(
                    visibility.syntax().clone(),
                    ast.visibility().unwrap().syntax().clone(),
                );
            }

            builder.map_node(name.syntax().clone(), ast.name().unwrap().syntax().clone());

            if let Some(field_list) = field_list {
                builder.map_node(
                    field_list.syntax().clone(),
                    ast.field_list().unwrap().syntax().clone(),
                );
            }

            if let Some(discriminant) = discriminant {
                builder
                    .map_node(discriminant.syntax().clone(), ast.expr().unwrap().syntax().clone());
            }

            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn token_tree(
        &self,
        delimiter: SyntaxKind,
        tt: Vec<NodeOrToken<ast::TokenTree, SyntaxToken>>,
    ) -> ast::TokenTree {
        let tt: Vec<_> = tt.into_iter().collect();
        let input: Vec<_> = tt.iter().cloned().filter_map(only_nodes).collect();

        let ast = make::token_tree(delimiter, tt).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(
                input.into_iter(),
                ast.token_trees_and_tokens().filter_map(only_nodes),
            );
            builder.finish(&mut mapping);
        }

        return ast;

        fn only_nodes(element: NodeOrToken<ast::TokenTree, SyntaxToken>) -> Option<SyntaxNode> {
            element.as_node().map(|it| it.syntax().clone())
        }
    }

    pub fn token(&self, kind: SyntaxKind) -> SyntaxToken {
        make::token(kind)
    }

    pub fn whitespace(&self, text: &str) -> ast::SyntaxToken {
        make::tokens::whitespace(text)
    }
}
