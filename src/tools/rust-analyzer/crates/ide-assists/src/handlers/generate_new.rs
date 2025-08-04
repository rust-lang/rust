use ide_db::{
    imports::import_assets::item_for_path_search, syntax_helpers::suggest_name::NameGenerator,
    use_trivial_constructor::use_trivial_constructor,
};
use syntax::{
    ast::{self, AstNode, HasName, HasVisibility, StructKind, edit_in_place::Indent, make},
    syntax_editor::Position,
};

use crate::{
    AssistContext, AssistId, Assists,
    utils::{find_struct_impl, generate_impl_with_item},
};

// Assist: generate_new
//
// Adds a `fn new` for a type.
//
// ```
// struct Ctx<T: Clone> {
//      data: T,$0
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//      data: T,
// }
//
// impl<T: Clone> Ctx<T> {
//     fn $0new(data: T) -> Self {
//         Self { data }
//     }
// }
// ```
pub(crate) fn generate_new(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;

    let field_list = match strukt.kind() {
        StructKind::Record(named) => {
            named.fields().filter_map(|f| Some((f.name()?, f.ty()?))).collect::<Vec<_>>()
        }
        StructKind::Tuple(tuple) => {
            let mut name_generator = NameGenerator::default();
            tuple
                .fields()
                .enumerate()
                .filter_map(|(i, f)| {
                    let ty = f.ty()?;
                    let name = match name_generator.for_type(
                        &ctx.sema.resolve_type(&ty)?,
                        ctx.db(),
                        ctx.edition(),
                    ) {
                        Some(name) => name,
                        None => name_generator.suggest_name(&format!("_{i}")),
                    };
                    Some((make::name(name.as_str()), f.ty()?))
                })
                .collect::<Vec<_>>()
        }
        StructKind::Unit => return None,
    };

    // Return early if we've found an existing new fn
    let impl_def =
        find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), &[String::from("new")])?;

    let current_module = ctx.sema.scope(strukt.syntax())?.module();

    let target = strukt.syntax().text_range();
    acc.add(AssistId::generate("generate_new"), "Generate `new`", target, |builder| {
        let trivial_constructors = field_list
            .iter()
            .map(|(name, ty)| {
                let ty = ctx.sema.resolve_type(ty)?;

                let item_in_ns = hir::ItemInNs::from(hir::ModuleDef::from(ty.as_adt()?));

                let type_path = current_module.find_path(
                    ctx.sema.db,
                    item_for_path_search(ctx.sema.db, item_in_ns)?,
                    ctx.config.import_path_config(),
                )?;

                let edition = current_module.krate().edition(ctx.db());

                let expr = use_trivial_constructor(
                    ctx.sema.db,
                    ide_db::helpers::mod_path_to_ast(&type_path, edition),
                    &ty,
                    edition,
                )?;

                Some((make::name_ref(&name.text()), Some(expr)))
            })
            .collect::<Vec<_>>();

        let params = field_list.iter().enumerate().filter_map(|(i, (name, ty))| {
            if trivial_constructors[i].is_none() {
                Some(make::param(make::ident_pat(false, false, name.clone()).into(), ty.clone()))
            } else {
                None
            }
        });
        let params = make::param_list(None, params);

        let fields = field_list.iter().enumerate().map(|(i, (name, _))| {
            if let Some(constructor) = trivial_constructors[i].clone() {
                constructor
            } else {
                (make::name_ref(&name.text()), None)
            }
        });

        let tail_expr: ast::Expr = match strukt.kind() {
            StructKind::Record(_) => {
                let fields = fields.map(|(name, expr)| make::record_expr_field(name, expr));
                let fields = make::record_expr_field_list(fields);
                make::record_expr(make::ext::ident_path("Self"), fields).into()
            }
            StructKind::Tuple(_) => {
                let args = fields.map(|(arg, expr)| {
                    let arg = || make::expr_path(make::path_unqualified(make::path_segment(arg)));
                    expr.unwrap_or_else(arg)
                });
                let arg_list = make::arg_list(args);
                make::expr_call(make::expr_path(make::ext::ident_path("Self")), arg_list).into()
            }
            StructKind::Unit => unreachable!(),
        };
        let body = make::block_expr(None, tail_expr.into());

        let ret_type = make::ret_type(make::ty_path(make::ext::ident_path("Self")));

        let fn_ = make::fn_(
            strukt.visibility(),
            make::name("new"),
            None,
            None,
            params,
            body,
            Some(ret_type),
            false,
            false,
            false,
            false,
        )
        .clone_for_update();
        fn_.indent(1.into());

        let mut editor = builder.make_editor(strukt.syntax());

        // Get the node for set annotation
        let contain_fn = if let Some(impl_def) = impl_def {
            fn_.indent(impl_def.indent_level());

            if let Some(l_curly) = impl_def.assoc_item_list().and_then(|list| list.l_curly_token())
            {
                editor.insert_all(
                    Position::after(l_curly),
                    vec![
                        make::tokens::whitespace(&format!("\n{}", impl_def.indent_level() + 1))
                            .into(),
                        fn_.syntax().clone().into(),
                        make::tokens::whitespace("\n").into(),
                    ],
                );
                fn_.syntax().clone()
            } else {
                let items = vec![ast::AssocItem::Fn(fn_)];
                let list = make::assoc_item_list(Some(items));
                editor.insert(Position::after(impl_def.syntax()), list.syntax());
                list.syntax().clone()
            }
        } else {
            // Generate a new impl to add the method to
            let indent_level = strukt.indent_level();
            let body = vec![ast::AssocItem::Fn(fn_)];
            let list = make::assoc_item_list(Some(body));
            let impl_def = generate_impl_with_item(&ast::Adt::Struct(strukt.clone()), Some(list));

            impl_def.indent(strukt.indent_level());

            // Insert it after the adt
            editor.insert_all(
                Position::after(strukt.syntax()),
                vec![
                    make::tokens::whitespace(&format!("\n\n{indent_level}")).into(),
                    impl_def.syntax().clone().into(),
                ],
            );
            impl_def.syntax().clone()
        };

        if let Some(fn_) = contain_fn.descendants().find_map(ast::Fn::cast)
            && let Some(cap) = ctx.config.snippet_cap
        {
            match strukt.kind() {
                StructKind::Tuple(_) => {
                    let struct_args = fn_
                        .body()
                        .unwrap()
                        .syntax()
                        .descendants()
                        .filter(|it| syntax::ast::ArgList::can_cast(it.kind()))
                        .flat_map(|args| args.children())
                        .filter(|it| syntax::ast::PathExpr::can_cast(it.kind()))
                        .enumerate()
                        .filter_map(|(i, node)| {
                            if trivial_constructors[i].is_none() { Some(node) } else { None }
                        });
                    if let Some(fn_params) = fn_.param_list() {
                        for (struct_arg, fn_param) in struct_args.zip(fn_params.params()) {
                            if let Some(fn_pat) = fn_param.pat() {
                                let fn_pat = fn_pat.syntax().clone();
                                let placeholder = builder.make_placeholder_snippet(cap);
                                editor.add_annotation_all(vec![struct_arg, fn_pat], placeholder)
                            }
                        }
                    }
                }
                _ => {}
            }

            // Add a tabstop before the name
            if let Some(name) = fn_.name() {
                let tabstop_before = builder.make_tabstop_before(cap);
                editor.add_annotation(name.syntax(), tabstop_before);
            }
        }

        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

#[cfg(test)]
mod record_tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_generate_new_with_zst_fields() {
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo { empty: Empty $0}
"#,
            r#"
struct Empty;

struct Foo { empty: Empty }

impl Foo {
    fn $0new() -> Self {
        Self { empty: Empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo { baz: String, empty: Empty $0}
"#,
            r#"
struct Empty;

struct Foo { baz: String, empty: Empty }

impl Foo {
    fn $0new(baz: String) -> Self {
        Self { baz, empty: Empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar }

struct Foo { empty: Empty $0}
"#,
            r#"
enum Empty { Bar }

struct Foo { empty: Empty }

impl Foo {
    fn $0new() -> Self {
        Self { empty: Empty::Bar }
    }
}
"#,
        );

        // make sure the assist only works on unit variants
        check_assist(
            generate_new,
            r#"
struct Empty {}

struct Foo { empty: Empty $0}
"#,
            r#"
struct Empty {}

struct Foo { empty: Empty }

impl Foo {
    fn $0new(empty: Empty) -> Self {
        Self { empty }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar {} }

struct Foo { empty: Empty $0}
"#,
            r#"
enum Empty { Bar {} }

struct Foo { empty: Empty }

impl Foo {
    fn $0new(empty: Empty) -> Self {
        Self { empty }
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_new() {
        check_assist(
            generate_new,
            r#"
struct Foo {$0}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<T: Clone> {$0}
"#,
            r#"
struct Foo<T: Clone> {}

impl<T: Clone> Foo<T> {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<'a, T: Foo<'a>> {$0}
"#,
            r#"
struct Foo<'a, T: Foo<'a>> {}

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo { baz: String $0}
"#,
            r#"
struct Foo { baz: String }

impl Foo {
    fn $0new(baz: String) -> Self {
        Self { baz }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo { baz: String, qux: Vec<i32> $0}
"#,
            r#"
struct Foo { baz: String, qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self {
        Self { baz, qux }
    }
}
"#,
        );
    }

    #[test]
    fn check_that_visibility_modifiers_dont_get_brought_in() {
        check_assist(
            generate_new,
            r#"
struct Foo { pub baz: String, pub qux: Vec<i32> $0}
"#,
            r#"
struct Foo { pub baz: String, pub qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self {
        Self { baz, qux }
    }
}
"#,
        );
    }

    #[test]
    fn check_it_reuses_existing_impls() {
        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn qux(&self) {}
}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }

    fn qux(&self) {}
}
"#,
        );

        check_assist(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
"#,
            r#"
struct Foo {}

impl Foo {
    fn $0new() -> Self {
        Self {  }
    }

    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
"#,
        );
    }

    #[test]
    fn non_zero_indent() {
        check_assist(
            generate_new,
            r#"
mod foo {
    struct $0Foo {}
}
"#,
            r#"
mod foo {
    struct Foo {}

    impl Foo {
        fn $0new() -> Self {
            Self {  }
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
        struct $0Foo {}
    }
}
"#,
            r#"
mod foo {
    mod bar {
        struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    struct $0Foo {}

    impl Foo {
        fn some() {}
    }
}
"#,
            r#"
mod foo {
    struct Foo {}

    impl Foo {
        fn $0new() -> Self {
            Self {  }
        }

        fn some() {}
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
        struct $0Foo {}

        impl Foo {
            fn some() {}
        }
    }
}
"#,
            r#"
mod foo {
    mod bar {
        struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }

            fn some() {}
        }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
mod foo {
    mod bar {
struct $0Foo {}

        impl Foo {
            fn some() {}
        }
    }
}
"#,
            r#"
mod foo {
    mod bar {
struct Foo {}

        impl Foo {
            fn $0new() -> Self {
                Self {  }
            }

            fn some() {}
        }
    }
}
"#,
        );
    }

    #[test]
    fn check_visibility_of_new_fn_based_on_struct() {
        check_assist(
            generate_new,
            r#"
pub struct Foo {$0}
"#,
            r#"
pub struct Foo {}

impl Foo {
    pub fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
pub(crate) struct Foo {$0}
"#,
            r#"
pub(crate) struct Foo {}

impl Foo {
    pub(crate) fn $0new() -> Self {
        Self {  }
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn new() -> Self {
        Self
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo {$0}

impl Foo {
    fn New() -> Self {
        Self
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_target() {
        check_assist_target(
            generate_new,
            r#"
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {$0}
struct EvenMoreIrrelevant;
"#,
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn test_unrelated_new() {
        check_assist(
            generate_new,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T> {
    pub file_id: HirFileId,$0
    pub ast: T,
}

impl<T> Source<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"#,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T> {
    pub file_id: HirFileId,
    pub ast: T,
}

impl<T> Source<T> {
    pub fn $0new(file_id: HirFileId, ast: T) -> Self {
        Self { file_id, ast }
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"#,
        );
    }
}

#[cfg(test)]
mod tuple_tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_generate_new_with_zst_fields() {
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo(Empty$0);
"#,
            r#"
struct Empty;

struct Foo(Empty);

impl Foo {
    fn $0new() -> Self {
        Self(Empty)
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Empty;

struct Foo(String, Empty$0);
"#,
            r#"
struct Empty;

struct Foo(String, Empty);

impl Foo {
    fn $0new(${1:_0}: String) -> Self {
        Self(${1:_0}, Empty)
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar }

struct Foo(Empty$0);
"#,
            r#"
enum Empty { Bar }

struct Foo(Empty);

impl Foo {
    fn $0new() -> Self {
        Self(Empty::Bar)
    }
}
"#,
        );

        // make sure the assist only works on unit variants
        check_assist(
            generate_new,
            r#"
struct Empty {}

struct Foo(Empty$0);
"#,
            r#"
struct Empty {}

struct Foo(Empty);

impl Foo {
    fn $0new(${1:empty}: Empty) -> Self {
        Self(${1:empty})
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
enum Empty { Bar {} }

struct Foo(Empty$0);
"#,
            r#"
enum Empty { Bar {} }

struct Foo(Empty);

impl Foo {
    fn $0new(${1:empty}: Empty) -> Self {
        Self(${1:empty})
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_new() {
        check_assist(
            generate_new,
            r#"
struct Foo($0);
"#,
            r#"
struct Foo();

impl Foo {
    fn $0new() -> Self {
        Self()
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<T: Clone>($0);
"#,
            r#"
struct Foo<T: Clone>();

impl<T: Clone> Foo<T> {
    fn $0new() -> Self {
        Self()
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo<'a, T: Foo<'a>>($0);
"#,
            r#"
struct Foo<'a, T: Foo<'a>>();

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn $0new() -> Self {
        Self()
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Foo(String$0);
"#,
            r#"
struct Foo(String);

impl Foo {
    fn $0new(${1:_0}: String) -> Self {
        Self(${1:_0})
    }
}
"#,
        );
        check_assist(
            generate_new,
            r#"
struct Vec<T> { };
struct Foo(String, Vec<i32>$0);
"#,
            r#"
struct Vec<T> { };
struct Foo(String, Vec<i32>);

impl Foo {
    fn $0new(${1:_0}: String, ${2:items}: Vec<i32>) -> Self {
        Self(${1:_0}, ${2:items})
    }
}
"#,
        );
    }

    #[test]
    fn check_that_visibility_modifiers_dont_get_brought_in() {
        check_assist(
            generate_new,
            r#"
struct Vec<T> { };
struct Foo(pub String, pub Vec<i32>$0);
"#,
            r#"
struct Vec<T> { };
struct Foo(pub String, pub Vec<i32>);

impl Foo {
    fn $0new(${1:_0}: String, ${2:items}: Vec<i32>) -> Self {
        Self(${1:_0}, ${2:items})
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo($0);

impl Foo {
    fn new() -> Self {
        Self
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_new,
            r#"
struct Foo($0);

impl Foo {
    fn New() -> Self {
        Self
    }
}
"#,
        );
    }

    #[test]
    fn generate_new_target() {
        check_assist_target(
            generate_new,
            r#"
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>>($0);
struct EvenMoreIrrelevant;
"#,
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>>();",
        );
    }

    #[test]
    fn test_unrelated_new() {
        check_assist(
            generate_new,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T>(pub HirFileId,$0 pub T);

impl<T> Source<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source(self.file_id, f(self.ast))
    }
}
"#,
            r#"
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }
}

pub struct Source<T>(pub HirFileId, pub T);

impl<T> Source<T> {
    pub fn $0new(${1:_0}: HirFileId, ${2:_1}: T) -> Self {
        Self(${1:_0}, ${2:_1})
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source(self.file_id, f(self.ast))
    }
}
"#,
        );
    }
}
