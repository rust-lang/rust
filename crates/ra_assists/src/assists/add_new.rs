use format_buf::format;
use hir::{db::HirDatabase, FromSource, InFile};
use join_to_string::join;
use ra_syntax::{
    ast::{
        self, AstNode, NameOwner, StructKind, TypeAscriptionOwner, TypeParamsOwner, VisibilityOwner,
    },
    TextUnit, T,
};
use std::fmt::Write;

use crate::{Assist, AssistCtx, AssistId};

// Assist: add_new
//
// Adds a new inherent impl for a type.
//
// ```
// struct Ctx<T: Clone> {
//      data: T,<|>
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//      data: T,
// }
//
// impl<T: Clone> Ctx<T> {
//     fn new(data: T) -> Self { Self { data } }
// }
//
// ```
pub(crate) fn add_new(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let strukt = ctx.find_node_at_offset::<ast::StructDef>()?;

    // We want to only apply this to non-union structs with named fields
    let field_list = match strukt.kind() {
        StructKind::Record(named) => named,
        _ => return None,
    };

    // Return early if we've found an existing new fn
    let impl_block = find_struct_impl(&ctx, &strukt)?;

    ctx.add_assist(AssistId("add_new"), "add new fn", |edit| {
        edit.target(strukt.syntax().text_range());

        let mut buf = String::with_capacity(512);

        if impl_block.is_some() {
            buf.push('\n');
        }

        let vis = strukt.visibility().map(|v| format!("{} ", v.syntax()));
        let vis = vis.as_ref().map(String::as_str).unwrap_or("");
        write!(&mut buf, "    {}fn new(", vis).unwrap();

        join(field_list.fields().filter_map(|f| {
            Some(format!("{}: {}", f.name()?.syntax().text(), f.ascribed_type()?.syntax().text()))
        }))
        .separator(", ")
        .to_buf(&mut buf);

        buf.push_str(") -> Self { Self {");

        join(field_list.fields().filter_map(|f| Some(f.name()?.syntax().text())))
            .separator(", ")
            .surround_with(" ", " ")
            .to_buf(&mut buf);

        buf.push_str("} }");

        let (start_offset, end_offset) = impl_block
            .and_then(|impl_block| {
                buf.push('\n');
                let start = impl_block
                    .syntax()
                    .descendants_with_tokens()
                    .find(|t| t.kind() == T!['{'])?
                    .text_range()
                    .end();

                Some((start, TextUnit::from_usize(1)))
            })
            .unwrap_or_else(|| {
                buf = generate_impl_text(&strukt, &buf);
                let start = strukt.syntax().text_range().end();

                (start, TextUnit::from_usize(3))
            });

        edit.set_cursor(start_offset + TextUnit::of_str(&buf) - end_offset);
        edit.insert(start_offset, buf);
    })
}

// Generates the surrounding `impl Type { <code> }` including type and lifetime
// parameters
fn generate_impl_text(strukt: &ast::StructDef, code: &str) -> String {
    let type_params = strukt.type_param_list();
    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\nimpl");
    if let Some(type_params) = &type_params {
        format!(buf, "{}", type_params.syntax());
    }
    buf.push_str(" ");
    buf.push_str(strukt.name().unwrap().text().as_str());
    if let Some(type_params) = type_params {
        let lifetime_params = type_params
            .lifetime_params()
            .filter_map(|it| it.lifetime_token())
            .map(|it| it.text().clone());
        let type_params =
            type_params.type_params().filter_map(|it| it.name()).map(|it| it.text().clone());
        join(lifetime_params.chain(type_params)).surround_with("<", ">").to_buf(&mut buf);
    }

    format!(&mut buf, " {{\n{}\n}}\n", code);

    buf
}

// Uses a syntax-driven approach to find any impl blocks for the struct that
// exist within the module/file
//
// Returns `None` if we've found an existing `new` fn
//
// FIXME: change the new fn checking to a more semantic approach when that's more
// viable (e.g. we process proc macros, etc)
fn find_struct_impl(
    ctx: &AssistCtx<impl HirDatabase>,
    strukt: &ast::StructDef,
) -> Option<Option<ast::ImplBlock>> {
    let db = ctx.db;
    let module = strukt.syntax().ancestors().find(|node| {
        ast::Module::can_cast(node.kind()) || ast::SourceFile::can_cast(node.kind())
    })?;

    let struct_ty = {
        let src = InFile { file_id: ctx.frange.file_id.into(), value: strukt.clone() };
        hir::Struct::from_source(db, src)?.ty(db)
    };

    let block = module.descendants().filter_map(ast::ImplBlock::cast).find_map(|impl_blk| {
        let src = InFile { file_id: ctx.frange.file_id.into(), value: impl_blk.clone() };
        let blk = hir::ImplBlock::from_source(db, src)?;

        let same_ty = blk.target_ty(db) == struct_ty;
        let not_trait_impl = blk.target_trait(db).is_none();

        if !(same_ty && not_trait_impl) {
            None
        } else {
            Some(impl_blk)
        }
    });

    if let Some(ref impl_blk) = block {
        if has_new_fn(impl_blk) {
            return None;
        }
    }

    Some(block)
}

fn has_new_fn(imp: &ast::ImplBlock) -> bool {
    if let Some(il) = imp.item_list() {
        for item in il.impl_items() {
            if let ast::ImplItem::FnDef(f) = item {
                if let Some(name) = f.name() {
                    if name.text().eq_ignore_ascii_case("new") {
                        return true;
                    }
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_add_new() {
        // Check output of generation
        check_assist(
            add_new,
"struct Foo {<|>}",
"struct Foo {}

impl Foo {
    fn new() -> Self { Self {  } }<|>
}
",
        );
        check_assist(
            add_new,
"struct Foo<T: Clone> {<|>}",
"struct Foo<T: Clone> {}

impl<T: Clone> Foo<T> {
    fn new() -> Self { Self {  } }<|>
}
",
        );
        check_assist(
            add_new,
"struct Foo<'a, T: Foo<'a>> {<|>}",
"struct Foo<'a, T: Foo<'a>> {}

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn new() -> Self { Self {  } }<|>
}
",
        );
        check_assist(
            add_new,
"struct Foo { baz: String <|>}",
"struct Foo { baz: String }

impl Foo {
    fn new(baz: String) -> Self { Self { baz } }<|>
}
",
        );
        check_assist(
            add_new,
"struct Foo { baz: String, qux: Vec<i32> <|>}",
"struct Foo { baz: String, qux: Vec<i32> }

impl Foo {
    fn new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }<|>
}
",
        );

        // Check that visibility modifiers don't get brought in for fields
        check_assist(
            add_new,
"struct Foo { pub baz: String, pub qux: Vec<i32> <|>}",
"struct Foo { pub baz: String, pub qux: Vec<i32> }

impl Foo {
    fn new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }<|>
}
",
        );

        // Check that it reuses existing impls
        check_assist(
            add_new,
"struct Foo {<|>}

impl Foo {}
",
"struct Foo {}

impl Foo {
    fn new() -> Self { Self {  } }<|>
}
",
        );
        check_assist(
            add_new,
"struct Foo {<|>}

impl Foo {
    fn qux(&self) {}
}
",
"struct Foo {}

impl Foo {
    fn new() -> Self { Self {  } }<|>

    fn qux(&self) {}
}
",
        );

        check_assist(
            add_new,
"struct Foo {<|>}

impl Foo {
    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
",
"struct Foo {}

impl Foo {
    fn new() -> Self { Self {  } }<|>

    fn qux(&self) {}
    fn baz() -> i32 {
        5
    }
}
",
        );

        // Check visibility of new fn based on struct
        check_assist(
            add_new,
"pub struct Foo {<|>}",
"pub struct Foo {}

impl Foo {
    pub fn new() -> Self { Self {  } }<|>
}
",
        );
        check_assist(
            add_new,
"pub(crate) struct Foo {<|>}",
"pub(crate) struct Foo {}

impl Foo {
    pub(crate) fn new() -> Self { Self {  } }<|>
}
",
        );
    }

    #[test]
    fn add_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            add_new,
            "
struct Foo {<|>}

impl Foo {
    fn new() -> Self {
        Self
    }
}",
        );

        check_assist_not_applicable(
            add_new,
            "
struct Foo {<|>}

impl Foo {
    fn New() -> Self {
        Self
    }
}",
        );
    }

    #[test]
    fn add_new_target() {
        check_assist_target(
            add_new,
            "
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {<|>}
struct EvenMoreIrrelevant;
",
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn test_unrelated_new() {
        check_assist(
            add_new,
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
    pub file_id: HirFileId,<|>
    pub ast: T,
}

impl<T> Source<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"##,
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
    pub fn new(file_id: HirFileId, ast: T) -> Self { Self { file_id, ast } }<|>

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"##,
        );
    }
}
