use hir::{AsAssocItem, Semantics};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, T};

use crate::{
    goto_definition::goto_definition, navigation_target::TryToNav, FilePosition, NavigationTarget,
    RangeInfo,
};

// Feature: Go to Declaration
//
// Navigates to the declaration of an identifier.
//
// This is the same as `Go to Definition` with the following exceptions:
// - outline modules will navigate to the `mod name;` item declaration
// - trait assoc items will navigate to the assoc item of the trait declaration opposed to the trait impl
// - fields in patterns will navigate to the field declaration of the struct, union or variant
pub(crate) fn goto_declaration(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let original_token = file
        .token_at_offset(position.offset)
        .find(|it| matches!(it.kind(), IDENT | T![self] | T![super] | T![crate] | T![Self]))?;
    let range = original_token.text_range();
    let info: Vec<NavigationTarget> = sema
        .descend_into_macros(original_token)
        .iter()
        .filter_map(|token| {
            let parent = token.parent()?;
            let def = match_ast! {
                match parent {
                    ast::NameRef(name_ref) => match NameRefClass::classify(&sema, &name_ref)? {
                        NameRefClass::Definition(it) => Some(it),
                        NameRefClass::FieldShorthand { field_ref, .. } => return field_ref.try_to_nav(db),
                    },
                    ast::Name(name) => match NameClass::classify(&sema, &name)? {
                        NameClass::Definition(it) | NameClass::ConstReference(it) => Some(it),
                        NameClass::PatFieldShorthand { field_ref, .. } => return field_ref.try_to_nav(db),
                    },
                    _ => None
                }
            };
            let assoc = match def? {
                Definition::Module(module) => {
                    return Some(NavigationTarget::from_module_to_decl(db, module))
                }
                Definition::Const(c) => c.as_assoc_item(db),
                Definition::TypeAlias(ta) => ta.as_assoc_item(db),
                Definition::Function(f) => f.as_assoc_item(db),
                _ => None,
            }?;

            let trait_ = assoc.containing_trait_impl(db)?;
            let name = Some(assoc.name(db)?);
            let item = trait_.items(db).into_iter().find(|it| it.name(db) == name)?;
            item.try_to_nav(db)
        })
        .collect();

    if info.is_empty() {
        goto_definition(db, position)
    } else {
        Some(RangeInfo::new(range, info))
    }
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;
    use itertools::Itertools;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis
            .goto_declaration(position)
            .unwrap()
            .expect("no declaration or definition found")
            .info;
        if navs.is_empty() {
            panic!("unresolved reference")
        }

        let cmp = |&FileRange { file_id, range }: &_| (file_id, range.start());
        let navs = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|(FileRange { file_id, range }, _)| FileRange { file_id, range })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        assert_eq!(expected, navs);
    }

    #[test]
    fn goto_decl_module_outline() {
        check(
            r#"
//- /main.rs
mod foo;
 // ^^^
//- /foo.rs
use self$0;
"#,
        )
    }

    #[test]
    fn goto_decl_module_inline() {
        check(
            r#"
mod foo {
 // ^^^
    use self$0;
}
"#,
        )
    }

    #[test]
    fn goto_decl_goto_def_fallback() {
        check(
            r#"
struct Foo;
    // ^^^
impl Foo$0 {}
"#,
        );
    }

    #[test]
    fn goto_decl_assoc_item_no_impl_item() {
        check(
            r#"
trait Trait {
    const C: () = ();
       // ^
}
impl Trait for () {}

fn main() {
    <()>::C$0;
}
"#,
        );
    }

    #[test]
    fn goto_decl_assoc_item() {
        check(
            r#"
trait Trait {
    const C: () = ();
       // ^
}
impl Trait for () {
    const C: () = ();
}

fn main() {
    <()>::C$0;
}
"#,
        );
        check(
            r#"
trait Trait {
    const C: () = ();
       // ^
}
impl Trait for () {
    const C$0: () = ();
}
"#,
        );
    }

    #[test]
    fn goto_decl_field_pat_shorthand() {
        check(
            r#"
struct Foo { field: u32 }
           //^^^^^
fn main() {
    let Foo { field$0 };
}
"#,
        );
    }

    #[test]
    fn goto_decl_constructor_shorthand() {
        check(
            r#"
struct Foo { field: u32 }
           //^^^^^
fn main() {
    let field = 0;
    Foo { field$0 };
}
"#,
        );
    }
}
