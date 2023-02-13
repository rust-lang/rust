//! A set of utils methods to reuse on other abstraction levels

use itertools::Itertools;

use crate::{ast, match_ast, AstNode, SyntaxKind};

pub fn path_to_string_stripping_turbo_fish(path: &ast::Path) -> String {
    path.syntax()
        .children()
        .filter_map(|node| {
            match_ast! {
                match node {
                    ast::PathSegment(it) => {
                        Some(it.name_ref()?.to_string())
                    },
                    ast::Path(it) => {
                        Some(path_to_string_stripping_turbo_fish(&it))
                    },
                    _ => None,
                }
            }
        })
        .join("::")
}

pub fn is_raw_identifier(name: &str) -> bool {
    let is_keyword = SyntaxKind::from_keyword(name).is_some();
    is_keyword && !matches!(name, "self" | "crate" | "super" | "Self")
}

#[cfg(test)]
mod tests {
    use super::path_to_string_stripping_turbo_fish;
    use crate::ast::make;

    #[test]
    fn turbofishes_are_stripped() {
        assert_eq!("Vec", path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::<i32>")),);
        assert_eq!(
            "Vec::new",
            path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::<i32>::new")),
        );
        assert_eq!(
            "Vec::new",
            path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::new()")),
        );
    }
}
