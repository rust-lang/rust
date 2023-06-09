use hir::{AsAssocItem, AssocItemContainer, HasCrate, HasSource};
use ide_db::{assists::AssistId, base_db::FileRange, defs::Definition, search::SearchScope};
use syntax::{
    ast::{self, edit::IndentLevel, edit_in_place::Indent, AstNode},
    SyntaxKind,
};

use crate::assist_context::{AssistContext, Assists};

// NOTE: Code may break if the self type implements a trait that has associated const with the same
// name, but it's pretty expensive to check that (`hir::Impl::all_for_type()`) and we assume that's
// pretty rare case.

// Assist: move_const_to_impl
//
// Move a local constant item in a method to impl's associated constant. All the references will be
// qualified with `Self::`.
//
// ```
// struct S;
// impl S {
//     fn foo() -> usize {
//         /// The answer.
//         const C$0: usize = 42;
//
//         C * C
//     }
// }
// ```
// ->
// ```
// struct S;
// impl S {
//     /// The answer.
//     const C: usize = 42;
//
//     fn foo() -> usize {
//         Self::C * Self::C
//     }
// }
// ```
pub(crate) fn move_const_to_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let db = ctx.db();
    let const_: ast::Const = ctx.find_node_at_offset()?;
    // Don't show the assist when the cursor is at the const's body.
    if let Some(body) = const_.body() {
        if body.syntax().text_range().contains(ctx.offset()) {
            return None;
        }
    }

    let parent_fn = const_.syntax().ancestors().find_map(ast::Fn::cast)?;

    // NOTE: We can technically provide this assist for default methods in trait definitions, but
    // it's somewhat complex to handle it correctly when the const's name conflicts with
    // supertrait's item. We may want to consider implementing it in the future.
    let AssocItemContainer::Impl(impl_) = ctx.sema.to_def(&parent_fn)?.as_assoc_item(db)?.container(db) else { return None; };
    if impl_.trait_(db).is_some() {
        return None;
    }

    let def = ctx.sema.to_def(&const_)?;
    let name = def.name(db)?;
    let items = impl_.source(db)?.value.assoc_item_list()?;

    let ty = impl_.self_ty(db);
    // If there exists another associated item with the same name, skip the assist.
    if ty
        .iterate_assoc_items(db, ty.krate(db), |assoc| {
            // Type aliases wouldn't conflict due to different namespaces, but we're only checking
            // the items in inherent impls, so we assume `assoc` is never type alias for the sake
            // of brevity (inherent associated types exist in nightly Rust, but it's *very*
            // unstable and we don't support them either).
            assoc.name(db).filter(|it| it == &name)
        })
        .is_some()
    {
        return None;
    }

    let usages =
        Definition::Const(def).usages(&ctx.sema).in_scope(SearchScope::file_range(FileRange {
            file_id: ctx.file_id(),
            range: parent_fn.syntax().text_range(),
        }));

    acc.add(
        AssistId("move_const_to_impl", crate::AssistKind::RefactorRewrite),
        "Move const to impl block",
        const_.syntax().text_range(),
        |builder| {
            let range_to_delete = match const_.syntax().next_sibling_or_token() {
                Some(s) if matches!(s.kind(), SyntaxKind::WHITESPACE) => {
                    // Remove following whitespaces too.
                    const_.syntax().text_range().cover(s.text_range())
                }
                _ => const_.syntax().text_range(),
            };
            builder.delete(range_to_delete);

            let const_ref = format!("Self::{}", name.display(ctx.db()));
            for range in usages.all().file_ranges().map(|it| it.range) {
                builder.replace(range, const_ref.clone());
            }

            // Heuristically inserting the extracted const after the consecutive existing consts
            // from the beginning of assoc items. We assume there are no inherent assoc type as
            // above.
            let last_const =
                items.assoc_items().take_while(|it| matches!(it, ast::AssocItem::Const(_))).last();
            let insert_offset = match &last_const {
                Some(it) => it.syntax().text_range().end(),
                None => match items.l_curly_token() {
                    Some(l_curly) => l_curly.text_range().end(),
                    // Not sure if this branch is ever reachable, but it wouldn't hurt to have a
                    // fallback.
                    None => items.syntax().text_range().start(),
                },
            };

            // If the moved const will be the first item of the impl, add a new line after that.
            //
            // We're assuming the code is formatted according to Rust's standard style guidelines
            // (i.e. no empty lines between impl's `{` token and its first assoc item).
            let fixup = if last_const.is_none() { "\n" } else { "" };
            let indent = IndentLevel::from_node(parent_fn.syntax());

            let const_ = const_.clone_for_update();
            const_.reindent_to(indent);
            builder.insert(insert_offset, format!("\n{indent}{const_}{fixup}"));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_to_top_level_const() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
const C$0: () = ();
"#,
        );
    }

    #[test]
    fn not_applicable_to_free_fn() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
fn f() {
    const C$0: () = ();
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_at_const_body() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() {
        const C: () = ($0);
    }
}
            "#,
        );
    }

    #[test]
    fn not_applicable_when_inside_const_body_block() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() {
        const C: () = {
            ($0)
        };
    }
}
            "#,
        );
    }

    #[test]
    fn not_applicable_to_trait_impl_fn() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
trait Trait {
    fn f();
}
impl Trait for () {
    fn f() {
        const C$0: () = ();
    }
}
"#,
        );
    }

    #[test]
    fn not_applicable_to_non_assoc_fn_inside_impl() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() {
        fn g() {
            const C$0: () = ();
        }
    }
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_const_with_same_name_exists() {
        check_assist_not_applicable(
            move_const_to_impl,
            r#"
struct S;
impl S {
    const C: usize = 42;
    fn f() {
        const C$0: () = ();
    }
"#,
        );

        check_assist_not_applicable(
            move_const_to_impl,
            r#"
struct S;
impl S {
    const C: usize = 42;
}
impl S {
    fn f() {
        const C$0: () = ();
    }
"#,
        );
    }

    #[test]
    fn move_const_simple_body() {
        check_assist(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() -> usize {
        /// doc comment
        const C$0: usize = 42;

        C * C
    }
}
"#,
            r#"
struct S;
impl S {
    /// doc comment
    const C: usize = 42;

    fn f() -> usize {
        Self::C * Self::C
    }
}
"#,
        );
    }

    #[test]
    fn move_const_simple_body_existing_const() {
        check_assist(
            move_const_to_impl,
            r#"
struct S;
impl S {
    const X: () = ();
    const Y: () = ();

    fn f() -> usize {
        /// doc comment
        const C$0: usize = 42;

        C * C
    }
}
"#,
            r#"
struct S;
impl S {
    const X: () = ();
    const Y: () = ();
    /// doc comment
    const C: usize = 42;

    fn f() -> usize {
        Self::C * Self::C
    }
}
"#,
        );
    }

    #[test]
    fn move_const_block_body() {
        check_assist(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() -> usize {
        /// doc comment
        const C$0: usize = {
            let a = 3;
            let b = 4;
            a * b
        };

        C * C
    }
}
"#,
            r#"
struct S;
impl S {
    /// doc comment
    const C: usize = {
        let a = 3;
        let b = 4;
        a * b
    };

    fn f() -> usize {
        Self::C * Self::C
    }
}
"#,
        );
    }

    #[test]
    fn correct_indent_when_nested() {
        check_assist(
            move_const_to_impl,
            r#"
fn main() {
    struct S;
    impl S {
        fn f() -> usize {
            /// doc comment
            const C$0: usize = 42;

            C * C
        }
    }
}
"#,
            r#"
fn main() {
    struct S;
    impl S {
        /// doc comment
        const C: usize = 42;

        fn f() -> usize {
            Self::C * Self::C
        }
    }
}
"#,
        )
    }

    #[test]
    fn move_const_in_nested_scope_with_same_name_in_other_scope() {
        check_assist(
            move_const_to_impl,
            r#"
struct S;
impl S {
    fn f() -> usize {
        const C: &str = "outer";

        let n = {
            /// doc comment
            const C$0: usize = 42;

            let m = {
                const C: &str = "inner";
                C.len()
            };

            C * m
        };

        n + C.len()
    }
}
"#,
            r#"
struct S;
impl S {
    /// doc comment
    const C: usize = 42;

    fn f() -> usize {
        const C: &str = "outer";

        let n = {
            let m = {
                const C: &str = "inner";
                C.len()
            };

            Self::C * m
        };

        n + C.len()
    }
}
"#,
        );
    }
}
