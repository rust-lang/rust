use ide_db::{
    imports::import_assets::item_for_path_search, use_trivial_constructor::use_trivial_constructor,
};
use itertools::Itertools;
use stdx::format_to;
use syntax::ast::{self, AstNode, HasName, HasVisibility, StructKind};

use crate::{
    utils::{find_impl_block_start, find_struct_impl, generate_impl_text},
    AssistContext, AssistId, AssistKind, Assists,
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
//     fn $0new(data: T) -> Self { Self { data } }
// }
// ```
pub(crate) fn generate_new(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;

    // We want to only apply this to non-union structs with named fields
    let field_list = match strukt.kind() {
        StructKind::Record(named) => named,
        _ => return None,
    };

    // Return early if we've found an existing new fn
    let impl_def =
        find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), &[String::from("new")])?;

    let current_module = ctx.sema.scope(strukt.syntax())?.module();

    let target = strukt.syntax().text_range();
    acc.add(AssistId("generate_new", AssistKind::Generate), "Generate `new`", target, |builder| {
        let mut buf = String::with_capacity(512);

        if impl_def.is_some() {
            buf.push('\n');
        }

        let vis = strukt.visibility().map_or(String::new(), |v| format!("{v} "));

        let trivial_constructors = field_list
            .fields()
            .map(|f| {
                let name = f.name()?;

                let ty = ctx.sema.resolve_type(&f.ty()?)?;

                let item_in_ns = hir::ItemInNs::from(hir::ModuleDef::from(ty.as_adt()?));

                let type_path = current_module.find_use_path(
                    ctx.sema.db,
                    item_for_path_search(ctx.sema.db, item_in_ns)?,
                    ctx.config.prefer_no_std,
                )?;

                let expr = use_trivial_constructor(
                    ctx.sema.db,
                    ide_db::helpers::mod_path_to_ast(&type_path),
                    &ty,
                )?;

                Some(format!("{name}: {expr}"))
            })
            .collect::<Vec<_>>();

        let params = field_list
            .fields()
            .enumerate()
            .filter_map(|(i, f)| {
                if trivial_constructors[i].is_none() {
                    let name = f.name()?;
                    let ty = f.ty()?;

                    Some(format!("{name}: {ty}"))
                } else {
                    None
                }
            })
            .format(", ");

        let fields = field_list
            .fields()
            .enumerate()
            .filter_map(|(i, f)| {
                let constructor = trivial_constructors[i].clone();
                if constructor.is_some() {
                    constructor
                } else {
                    Some(f.name()?.to_string())
                }
            })
            .format(", ");

        format_to!(buf, "    {vis}fn new({params}) -> Self {{ Self {{ {fields} }} }}");

        let start_offset = impl_def
            .and_then(|impl_def| find_impl_block_start(impl_def, &mut buf))
            .unwrap_or_else(|| {
                buf = generate_impl_text(&ast::Adt::Struct(strukt.clone()), &buf);
                strukt.syntax().text_range().end()
            });

        match ctx.config.snippet_cap {
            None => builder.insert(start_offset, buf),
            Some(cap) => {
                buf = buf.replace("fn new", "fn $0new");
                builder.insert_snippet(cap, start_offset, buf);
            }
        }
    })
}

#[cfg(test)]
mod tests {
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
    fn $0new() -> Self { Self { empty: Empty } }
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
    fn $0new(baz: String) -> Self { Self { baz, empty: Empty } }
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
    fn $0new() -> Self { Self { empty: Empty::Bar } }
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
    fn $0new(empty: Empty) -> Self { Self { empty } }
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
    fn $0new(empty: Empty) -> Self { Self { empty } }
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
    fn $0new() -> Self { Self {  } }
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
    fn $0new() -> Self { Self {  } }
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
    fn $0new() -> Self { Self {  } }
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
    fn $0new(baz: String) -> Self { Self { baz } }
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
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
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
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
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
    fn $0new() -> Self { Self {  } }
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
    fn $0new() -> Self { Self {  } }

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
    fn $0new() -> Self { Self {  } }

    fn qux(&self) {}
    fn baz() -> i32 {
        5
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
    pub fn $0new() -> Self { Self {  } }
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
    pub(crate) fn $0new() -> Self { Self {  } }
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
    pub fn $0new(file_id: HirFileId, ast: T) -> Self { Self { file_id, ast } }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"#,
        );
    }
}
