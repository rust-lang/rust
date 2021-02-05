use hir::Adt;
use stdx::format_to;
use syntax::ast::{self, AstNode, NameOwner};
use syntax::{ast::VisibilityOwner, T};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: generate_enum_match_method
//
// Generate an `is_` method for an enum variant.
//
// ```
// enum Version {
//  Undefined,
//  Minor$0,
//  Major,
// }
// ```
// ->
// ```
// enum Version {
//  Undefined,
//  Minor,
//  Major,
// }
//
// impl Version {
//     fn is_minor(&self) -> bool {
//         matches!(self, Self::Minor)
//     }
// }
// ```
pub(crate) fn generate_enum_match_method(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = variant.parent_enum();
    if !matches!(variant.kind(), ast::StructKind::Unit) {
        mark::hit!(test_gen_enum_match_on_non_unit_variant_not_implemented);
        return None;
    }

    let fn_name = to_lower_snake_case(&format!("{}", variant_name));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(&ctx, &parent_enum, format!("is_{}", fn_name).as_str())?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("generate_enum_match_method", AssistKind::Generate),
        "Generate an `is_` method for an enum variant",
        target,
        |builder| {
            let mut buf = String::with_capacity(512);

            if impl_def.is_some() {
                buf.push('\n');
            }

            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));

            format_to!(
                buf,
                "    {}fn is_{}(&self) -> bool {{
        matches!(self, Self::{})
    }}",
                vis,
                fn_name,
                variant_name
            );

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
                    buf = generate_impl_text(&parent_enum, &buf);
                    parent_enum.syntax().text_range().end()
                });

            builder.insert(start_offset, buf);
        },
    )
}

// Generates the surrounding `impl Type { <code> }` including type and lifetime
// parameters
fn generate_impl_text(strukt: &ast::Enum, code: &str) -> String {
    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\nimpl");
    buf.push_str(" ");
    buf.push_str(strukt.name().unwrap().text());
    format_to!(buf, " {{\n{}\n}}", code);
    buf
}

fn to_lower_snake_case(s: &str) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut prev = false;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev {
            buf.push('_')
        }
        prev = true;

        buf.push(c.to_ascii_lowercase());
    }
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
    ctx: &AssistContext,
    strukt: &ast::Enum,
    name: &str,
) -> Option<Option<ast::Impl>> {
    let db = ctx.db();
    let module = strukt.syntax().ancestors().find(|node| {
        ast::Module::can_cast(node.kind()) || ast::SourceFile::can_cast(node.kind())
    })?;

    let struct_def = ctx.sema.to_def(strukt)?;

    let block = module.descendants().filter_map(ast::Impl::cast).find_map(|impl_blk| {
        let blk = ctx.sema.to_def(&impl_blk)?;

        // FIXME: handle e.g. `struct S<T>; impl<U> S<U> {}`
        // (we currently use the wrong type parameter)
        // also we wouldn't want to use e.g. `impl S<u32>`
        let same_ty = match blk.target_ty(db).as_adt() {
            Some(def) => def == Adt::Enum(struct_def),
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
        if has_fn(impl_blk, name) {
            mark::hit!(test_gen_enum_match_impl_already_exists);
            return None;
        }
    }

    Some(block)
}

fn has_fn(imp: &ast::Impl, rhs_name: &str) -> bool {
    if let Some(il) = imp.assoc_item_list() {
        for item in il.assoc_items() {
            if let ast::AssocItem::Fn(f) = item {
                if let Some(name) = f.name() {
                    if name.text().eq_ignore_ascii_case(rhs_name) {
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
    use ide_db::helpers::FamousDefs;
    use test_utils::mark;

    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    fn check_not_applicable(ra_fixture: &str) {
        let fixture =
            format!("//- /main.rs crate:main deps:core\n{}\n{}", ra_fixture, FamousDefs::FIXTURE);
        check_assist_not_applicable(generate_enum_match_method, &fixture)
    }

    #[test]
    fn test_generate_enum_match_from_variant() {
        check_assist(
            generate_enum_match_method,
            r#"
enum Variant {
    Undefined,
    Minor$0,
    Major,
}"#,
            r#"enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Variant {
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_match_already_implemented() {
        mark::check!(test_gen_enum_match_impl_already_exists);
        check_not_applicable(
            r#"
enum Variant {
    Undefined,
    Minor$0,
    Major,
}

impl Variant {
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }

    #[test]
    fn test_add_from_impl_no_element() {
        mark::check!(test_gen_enum_match_on_non_unit_variant_not_implemented);
        check_not_applicable(
            r#"
enum Variant {
    Undefined,
    Minor(u32)$0,
    Major,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_match_from_variant_with_one_variant() {
        check_assist(
            generate_enum_match_method,
            r#"enum Variant { Undefi$0ned }"#,
            r#"
enum Variant { Undefined }

impl Variant {
    fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_match_from_variant_with_visibility_marker() {
        check_assist(
            generate_enum_match_method,
            r#"
pub(crate) enum Variant {
    Undefined,
    Minor$0,
    Major,
}"#,
            r#"pub(crate) enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Variant {
    pub(crate) fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }
}
