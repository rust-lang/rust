use hir::Adt;
use ra_syntax::{
    ast::{
        self, AstNode, NameOwner, StructKind, TypeAscriptionOwner, TypeParamsOwner, VisibilityOwner,
    },
    T,
};
use stdx::{format_to, SepBy};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: generate_new
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
//     fn $0new(data: T) -> Self { Self { data } }
// }
//
// ```
pub(crate) fn generate_new(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::StructDef>()?;

    // We want to only apply this to non-union structs with named fields
    let field_list = match strukt.kind() {
        StructKind::Record(named) => named,
        _ => return None,
    };

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(&ctx, &strukt)?;

    let target = strukt.syntax().text_range();
    acc.add(AssistId("generate_new", AssistKind::Generate), "Generate `new`", target, |builder| {
        let mut buf = String::with_capacity(512);

        if impl_def.is_some() {
            buf.push('\n');
        }

        let vis = strukt.visibility().map_or(String::new(), |v| format!("{} ", v));

        let params = field_list
            .fields()
            .filter_map(|f| {
                Some(format!("{}: {}", f.name()?.syntax(), f.ascribed_type()?.syntax()))
            })
            .sep_by(", ");
        let fields = field_list.fields().filter_map(|f| f.name()).sep_by(", ");

        format_to!(buf, "    {}fn new({}) -> Self {{ Self {{ {} }} }}", vis, params, fields);

        let start_offset = impl_def
            .and_then(|impl_def| {
                buf.push('\n');
                let start = impl_def
                    .syntax()
                    .descendants_with_tokens()
                    .find(|t| t.kind() == T!['{'])?
                    .text_range()
                    .end();

                Some(start)
            })
            .unwrap_or_else(|| {
                buf = generate_impl_text(&strukt, &buf);
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

// Generates the surrounding `impl Type { <code> }` including type and lifetime
// parameters
fn generate_impl_text(strukt: &ast::StructDef, code: &str) -> String {
    let type_params = strukt.type_param_list();
    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\nimpl");
    if let Some(type_params) = &type_params {
        format_to!(buf, "{}", type_params.syntax());
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
        format_to!(buf, "<{}>", lifetime_params.chain(type_params).sep_by(", "))
    }

    format_to!(buf, " {{\n{}\n}}\n", code);

    buf
}

// Uses a syntax-driven approach to find any impl blocks for the struct that
// exist within the module/file
//
// Returns `None` if we've found an existing `new` fn
//
// FIXME: change the new fn checking to a more semantic approach when that's more
// viable (e.g. we process proc macros, etc)
fn find_struct_impl(ctx: &AssistContext, strukt: &ast::StructDef) -> Option<Option<ast::ImplDef>> {
    let db = ctx.db();
    let module = strukt.syntax().ancestors().find(|node| {
        ast::Module::can_cast(node.kind()) || ast::SourceFile::can_cast(node.kind())
    })?;

    let struct_def = ctx.sema.to_def(strukt)?;

    let block = module.descendants().filter_map(ast::ImplDef::cast).find_map(|impl_blk| {
        let blk = ctx.sema.to_def(&impl_blk)?;

        // FIXME: handle e.g. `struct S<T>; impl<U> S<U> {}`
        // (we currently use the wrong type parameter)
        // also we wouldn't want to use e.g. `impl S<u32>`
        let same_ty = match blk.target_ty(db).as_adt() {
            Some(def) => def == Adt::Struct(struct_def),
            None => false,
        };
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

fn has_new_fn(imp: &ast::ImplDef) -> bool {
    if let Some(il) = imp.item_list() {
        for item in il.assoc_items() {
            if let ast::AssocItem::FnDef(f) = item {
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
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_generate_new() {
        // Check output of generation
        check_assist(
            generate_new,
"struct Foo {<|>}",
"struct Foo {}

impl Foo {
    fn $0new() -> Self { Self {  } }
}
",
        );
        check_assist(
            generate_new,
"struct Foo<T: Clone> {<|>}",
"struct Foo<T: Clone> {}

impl<T: Clone> Foo<T> {
    fn $0new() -> Self { Self {  } }
}
",
        );
        check_assist(
            generate_new,
"struct Foo<'a, T: Foo<'a>> {<|>}",
"struct Foo<'a, T: Foo<'a>> {}

impl<'a, T: Foo<'a>> Foo<'a, T> {
    fn $0new() -> Self { Self {  } }
}
",
        );
        check_assist(
            generate_new,
"struct Foo { baz: String <|>}",
"struct Foo { baz: String }

impl Foo {
    fn $0new(baz: String) -> Self { Self { baz } }
}
",
        );
        check_assist(
            generate_new,
"struct Foo { baz: String, qux: Vec<i32> <|>}",
"struct Foo { baz: String, qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
}
",
        );

        // Check that visibility modifiers don't get brought in for fields
        check_assist(
            generate_new,
"struct Foo { pub baz: String, pub qux: Vec<i32> <|>}",
"struct Foo { pub baz: String, pub qux: Vec<i32> }

impl Foo {
    fn $0new(baz: String, qux: Vec<i32>) -> Self { Self { baz, qux } }
}
",
        );

        // Check that it reuses existing impls
        check_assist(
            generate_new,
"struct Foo {<|>}

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
"struct Foo {<|>}

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
"pub struct Foo {<|>}",
"pub struct Foo {}

impl Foo {
    pub fn $0new() -> Self { Self {  } }
}
",
        );
        check_assist(
            generate_new,
"pub(crate) struct Foo {<|>}",
"pub(crate) struct Foo {}

impl Foo {
    pub(crate) fn $0new() -> Self { Self {  } }
}
",
        );
    }

    #[test]
    fn generate_new_not_applicable_if_fn_exists() {
        check_assist_not_applicable(
            generate_new,
            "
struct Foo {<|>}

impl Foo {
    fn new() -> Self {
        Self
    }
}",
        );

        check_assist_not_applicable(
            generate_new,
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
    fn generate_new_target() {
        check_assist_target(
            generate_new,
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
    pub fn $0new(file_id: HirFileId, ast: T) -> Self { Self { file_id, ast } }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
}
"##,
        );
    }
}
