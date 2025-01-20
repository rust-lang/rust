//! Wrappers over [`make`] constructors
use crate::{
    ast::{
        self, make, HasArgList, HasGenericArgs, HasGenericParams, HasName, HasTypeBounds,
        HasVisibility,
    },
    syntax_editor::SyntaxMappingBuilder,
    AstNode, NodeOrToken, SyntaxKind, SyntaxNode, SyntaxToken,
};

use super::SyntaxFactory;

impl SyntaxFactory {
    pub fn name(&self, name: &str) -> ast::Name {
        make::name(name).clone_for_update()
    }

    pub fn name_ref(&self, name: &str) -> ast::NameRef {
        make::name_ref(name).clone_for_update()
    }

    pub fn lifetime(&self, text: &str) -> ast::Lifetime {
        make::lifetime(text).clone_for_update()
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

    pub fn ty_path(&self, path: ast::Path) -> ast::PathType {
        let ast::Type::PathType(ast) = make::ty_path(path.clone()).clone_for_update() else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(path.syntax().clone(), ast.path().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

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

    pub fn path_segment(&self, name_ref: ast::NameRef) -> ast::PathSegment {
        let ast = make::path_segment(name_ref.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(name_ref.syntax().clone(), ast.name_ref().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn path_segment_generics(
        &self,
        name_ref: ast::NameRef,
        generic_arg_list: ast::GenericArgList,
    ) -> ast::PathSegment {
        let ast::Type::PathType(path) = make::ty(&format!("{name_ref}{generic_arg_list}")) else {
            unreachable!();
        };

        let ast = path.path().unwrap().segment().unwrap().clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(name_ref.syntax().clone(), ast.name_ref().unwrap().syntax().clone());
            builder.map_node(
                generic_arg_list.syntax().clone(),
                ast.generic_arg_list().unwrap().syntax().clone(),
            );
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn path_unqualified(&self, segment: ast::PathSegment) -> ast::Path {
        let ast = make::path_unqualified(segment.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(segment.syntax().clone(), ast.segment().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn path_from_segments(
        &self,
        segments: impl IntoIterator<Item = ast::PathSegment>,
        is_abs: bool,
    ) -> ast::Path {
        let (segments, input) = iterator_input(segments);
        let ast = make::path_from_segments(segments, is_abs).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(input.into_iter(), ast.segments().map(|it| it.syntax().clone()));
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

    pub fn wildcard_pat(&self) -> ast::WildcardPat {
        make::wildcard_pat().clone_for_update()
    }

    pub fn literal_pat(&self, text: &str) -> ast::LiteralPat {
        make::literal_pat(text).clone_for_update()
    }

    pub fn tuple_struct_pat(
        &self,
        path: ast::Path,
        fields: impl IntoIterator<Item = ast::Pat>,
    ) -> ast::TupleStructPat {
        let (fields, input) = iterator_input(fields);
        let ast = make::tuple_struct_pat(path.clone(), fields).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(path.syntax().clone(), ast.path().unwrap().syntax().clone());
            builder.map_children(input.into_iter(), ast.fields().map(|it| it.syntax().clone()));
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn block_expr(
        &self,
        statements: impl IntoIterator<Item = ast::Stmt>,
        tail_expr: Option<ast::Expr>,
    ) -> ast::BlockExpr {
        let (statements, mut input) = iterator_input(statements);

        let ast = make::block_expr(statements, tail_expr.clone()).clone_for_update();

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
        make::expr_empty_block().clone_for_update()
    }

    pub fn expr_tuple(&self, fields: impl IntoIterator<Item = ast::Expr>) -> ast::TupleExpr {
        let (fields, input) = iterator_input(fields);
        let ast = make::expr_tuple(fields).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(input.into_iter(), ast.fields().map(|it| it.syntax().clone()));
            builder.finish(&mut mapping);
        }

        ast
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

    pub fn expr_literal(&self, text: &str) -> ast::Literal {
        make::expr_literal(text).clone_for_update()
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

    pub fn expr_prefix(&self, op: SyntaxKind, expr: ast::Expr) -> ast::PrefixExpr {
        let ast = make::expr_prefix(op, expr.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_call(&self, expr: ast::Expr, arg_list: ast::ArgList) -> ast::CallExpr {
        // FIXME: `make::expr_call`` should return a `CallExpr`, not just an `Expr`
        let ast::Expr::CallExpr(ast) =
            make::expr_call(expr.clone(), arg_list.clone()).clone_for_update()
        else {
            unreachable!()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.map_node(arg_list.syntax().clone(), ast.arg_list().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn arg_list(&self, args: impl IntoIterator<Item = ast::Expr>) -> ast::ArgList {
        let (args, input) = iterator_input(args);
        let ast = make::arg_list(args).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax.clone());
            builder.map_children(input.into_iter(), ast.args().map(|it| it.syntax().clone()));
            builder.finish(&mut mapping);
        }

        ast
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

    pub fn expr_if(
        &self,
        condition: ast::Expr,
        then_branch: ast::BlockExpr,
        else_branch: Option<ast::ElseBranch>,
    ) -> ast::IfExpr {
        let ast = make::expr_if(condition.clone(), then_branch.clone(), else_branch.clone())
            .clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(condition.syntax().clone(), ast.condition().unwrap().syntax().clone());
            builder.map_node(
                then_branch.syntax().clone(),
                ast.then_branch().unwrap().syntax().clone(),
            );

            if let Some(else_branch) = else_branch {
                builder.map_node(
                    else_branch.syntax().clone(),
                    ast.else_branch().unwrap().syntax().clone(),
                );
            }
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_let(&self, pattern: ast::Pat, expr: ast::Expr) -> ast::LetExpr {
        let ast = make::expr_let(pattern.clone(), expr.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(pattern.syntax().clone(), ast.pat().unwrap().syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_stmt(&self, expr: ast::Expr) -> ast::ExprStmt {
        let ast = make::expr_stmt(expr.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn expr_match(&self, expr: ast::Expr, match_arm_list: ast::MatchArmList) -> ast::MatchExpr {
        let ast = make::expr_match(expr.clone(), match_arm_list.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.map_node(
                match_arm_list.syntax().clone(),
                ast.match_arm_list().unwrap().syntax().clone(),
            );
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn match_arm(
        &self,
        pat: ast::Pat,
        guard: Option<ast::MatchGuard>,
        expr: ast::Expr,
    ) -> ast::MatchArm {
        let ast = make::match_arm(pat.clone(), guard.clone(), expr.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(pat.syntax().clone(), ast.pat().unwrap().syntax().clone());
            if let Some(guard) = guard {
                builder.map_node(guard.syntax().clone(), ast.guard().unwrap().syntax().clone());
            }
            builder.map_node(expr.syntax().clone(), ast.expr().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn match_guard(&self, condition: ast::Expr) -> ast::MatchGuard {
        let ast = make::match_guard(condition.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(condition.syntax().clone(), ast.condition().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn match_arm_list(
        &self,
        match_arms: impl IntoIterator<Item = ast::MatchArm>,
    ) -> ast::MatchArmList {
        let (match_arms, input) = iterator_input(match_arms);
        let ast = make::match_arm_list(match_arms).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(input.into_iter(), ast.arms().map(|it| it.syntax().clone()));
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

    pub fn type_arg(&self, ty: ast::Type) -> ast::TypeArg {
        let ast = make::type_arg(ty.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(ty.syntax().clone(), ast.ty().unwrap().syntax().clone());
            builder.finish(&mut mapping);
        }

        ast
    }

    pub fn lifetime_arg(&self, lifetime: ast::Lifetime) -> ast::LifetimeArg {
        let ast = make::lifetime_arg(lifetime.clone()).clone_for_update();

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_node(lifetime.syntax().clone(), ast.lifetime().unwrap().syntax().clone());
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

    pub fn generic_arg_list(
        &self,
        generic_args: impl IntoIterator<Item = ast::GenericArg>,
        is_turbo: bool,
    ) -> ast::GenericArgList {
        let (generic_args, input) = iterator_input(generic_args);
        let ast = if is_turbo {
            make::turbofish_generic_arg_list(generic_args).clone_for_update()
        } else {
            make::generic_arg_list(generic_args).clone_for_update()
        };

        if let Some(mut mapping) = self.mappings() {
            let mut builder = SyntaxMappingBuilder::new(ast.syntax().clone());
            builder.map_children(
                input.into_iter(),
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
        let (fields, input) = iterator_input(fields);
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
        let (fields, input) = iterator_input(fields);
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
        let (variants, input) = iterator_input(variants);
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
        tt: impl IntoIterator<Item = NodeOrToken<ast::TokenTree, SyntaxToken>>,
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

    pub fn whitespace(&self, text: &str) -> SyntaxToken {
        make::tokens::whitespace(text)
    }
}

// `ext` constructors
impl SyntaxFactory {
    pub fn ident_path(&self, ident: &str) -> ast::Path {
        self.path_unqualified(self.path_segment(self.name_ref(ident)))
    }

    pub fn expr_unit(&self) -> ast::Expr {
        self.expr_tuple([]).into()
    }

    pub fn ty_option(&self, t: ast::Type) -> ast::PathType {
        let generic_arg_list = self.generic_arg_list([self.type_arg(t).into()], false);
        let path = self.path_unqualified(
            self.path_segment_generics(self.name_ref("Option"), generic_arg_list),
        );

        self.ty_path(path)
    }

    pub fn ty_result(&self, t: ast::Type, e: ast::Type) -> ast::PathType {
        let generic_arg_list =
            self.generic_arg_list([self.type_arg(t).into(), self.type_arg(e).into()], false);
        let path = self.path_unqualified(
            self.path_segment_generics(self.name_ref("Result"), generic_arg_list),
        );

        self.ty_path(path)
    }
}

// We need to collect `input` here instead of taking `impl IntoIterator + Clone`,
// because if we took `impl IntoIterator + Clone`, that could be something like an
// `Iterator::map` with a closure that also makes use of a `SyntaxFactory` constructor.
//
// In that case, the iterator would be evaluated inside of the call to `map_children`,
// and the inner constructor would try to take a mutable borrow of the mappings `RefCell`,
// which would panic since it's already being mutably borrowed in the outer constructor.
fn iterator_input<N: AstNode>(input: impl IntoIterator<Item = N>) -> (Vec<N>, Vec<SyntaxNode>) {
    input
        .into_iter()
        .map(|it| {
            let syntax = it.syntax().clone();
            (it, syntax)
        })
        .collect()
}
