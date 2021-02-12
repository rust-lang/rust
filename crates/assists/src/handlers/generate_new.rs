use ast::Adt;
use itertools::Itertools;
use stdx::format_to;
use syntax::ast::{self, AstNode, NameOwner, StructKind, VisibilityOwner};

use crate::{
    utils::{find_impl_block_start, find_struct_impl, generate_impl_text},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: generate_new
//
// Adds a new inherent impl for a type.
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
pub(crate) fn generate_new(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;

    // We want to only apply this to non-union structs with named fields
    let field_list = match strukt.kind() {
        StructKind::Record(named) => named,
        _ => return None,
    };

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(&ctx, &Adt::Struct(strukt.clone()), "new")?;

    let target = strukt.syntax().text_range();
    acc.add(AssistId("generate_new", AssistKind::Generate), "Generate `new`", target, |builder| {
        let mut buf = String::with_capacity(512);

        if impl_def.is_some() {
            buf.push('\n');
        }

        let vis = strukt.visibility().map_or(String::new(), |v| format!("{} ", v));

        let params = field_list
            .fields()
            .filter_map(|f| Some(format!("{}: {}", f.name()?.syntax(), f.ty()?.syntax())))
            .format(", ");
        let fields = field_list.fields().filter_map(|f| f.name()).format(", ");

        format_to!(buf, "    {}fn new({}) -> Self {{ Self {{ {} }} }}", vis, params, fields);

        let start_offset = impl_def
            .and_then(|impl_def| find_impl_block_start(impl_def, &mut buf))
            .unwrap_or_else(|| {
                buf = generate_impl_text(&Adt::Struct(strukt.clone()), &buf);
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
    #[rustfmt::skip]
    fn test_generate_new() {
        // Check output of generation
        check_assist(
            generate_new,
"struct Foo {$0}",
"struct Foo {}

impl Foo {
    fn $0new() -> Self { Self {  } }
}",
        );
        check_assist(
            generate_new,
"struct Foo<T: Clone> {$0}",
"struct Foo<T: Clone> {}

impl<T: Clone> Foo<T> {
    fn $0new() -> Self { Self {  } }
}",
        );
        check_assist(
            generate_new,
"struct Foo<'a, T: Foo<'a>> {$0}",
"struct Foo<'a, T: Foo<'a>> {}

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn $0new() -> Self { Self {  } }
}",
        );
        check_assist(
            generate_new,
"struct Foo { baz: String $0}",
"struct Foo { baz: String }

impl Foo {
    fn $0new(baz: String) -> Self { Self { baz } }
}",
        );
        check_assist(
            generate_new,
"struct Foo { baz: String, qux: Vec<i32> $0}",
"struct Foo { baz: String, qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
}",
        );

        // Check that visibility modifiers don't get brought in for fields
        check_assist(
            generate_new,
"struct Foo { pub baz: String, pub qux: Vec<i32> $0}",
"struct Foo { pub baz: String, pub qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
}",
        );

        // Check that it reuses existing impls
        check_assist(
            generate_new,
"struct Foo {$0}

impl Foo {}
",
"struct Foo {}

impl Foo {
    fn $0new() -> Self { Self {  } }
}
",
        );
        check_assist(
            generate_new,
"struct Foo {$0}

impl Foo {
    fn qux(&self) {}
}
",
"struct Foo {}

impl Foo {
    fn $0new() -> Self { Self {  } }

    fn qux(&self) {}
}
",
        );

        check_assist(
            generate_new,
"struct Foo {$0}

impl Foo {
    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
",
"struct Foo {}

impl Foo {
    fn $0new() -> Self { Self {  } }

    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
",
        );

        // Check visibility of new fn based on struct
        check_assist(
            generate_new,
"pub struct Foo {$0}",
"pub struct Foo {}

impl Foo {
    pub fn $0new() -> Self { Self {  } }
}",
        );
        check_assist(
            generate_new,
"pub(crate) struct Foo {$0}",
"pub(crate) struct Foo {}

impl Foo {
    pub(crate) fn $0new() -> Self { Self {  } }
}",
        );
    }

    #[test]
    fn generate_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            generate_new,
            "
struct Foo {$0}

impl Foo {
    fn new() -> Self {
        Self
    }
}",
        );

        check_assist_not_applicable(
            generate_new,
            "
struct Foo {$0}

impl Foo {
    fn New() -> Self {
        Self
    }
}",
        );
    }

    #[test]
    fn generate_new_target() {
        check_assist_target(
            generate_new,
            "
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {$0}
struct EvenMoreIrrelevant;
",
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn test_unrelated_new() {
        check_assist(
            generate_new,
            r##"
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
}"##,
            r##"
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
}"##,
        );
    }
}
