use libsyntax2::{
    File, TextUnit, AstNode,
    ast::self,
    algo::{
        ancestors,
    },
};

use {
    AtomEdit, find_node_at_offset,
    scope::FnScopes,
};

#[derive(Debug)]
pub struct CompletionItem {
    pub name: String,
}

pub fn scope_completion(file: &File, offset: TextUnit) -> Option<Vec<CompletionItem>> {
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        // Don't bother with completion if incremental reparse fails
        file.incremental_reparse(&edit)?
    };
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), offset)?;
    let fn_def = ancestors(name_ref.syntax()).filter_map(ast::FnDef::cast).next()?;
    let scopes = FnScopes::new(fn_def);
    Some(complete(name_ref, &scopes))
}

fn complete(name_ref: ast::NameRef, scopes: &FnScopes) -> Vec<CompletionItem> {
    scopes.scope_chain(name_ref.syntax())
        .flat_map(|scope| scopes.entries(scope).iter())
        .map(|entry| CompletionItem {
            name: entry.name().to_string()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::{assert_eq_dbg, extract_offset};

    fn do_check(code: &str, expected_completions: &str) {
        let (off, code) = extract_offset(&code);
        let file = File::parse(&code);
        let completions = scope_completion(&file, off).unwrap();
        assert_eq_dbg(expected_completions, &completions);
    }

    #[test]
    fn test_completion_let_scope() {
        do_check(r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ", r#"[CompletionItem { name: "y" },
                   CompletionItem { name: "x" }]"#);
    }

    #[test]
    fn test_completion_if_let_scope() {
        do_check(r"
            fn quux() {
                if let Some(x) = foo() {
                    let y = 92;
                };
                if let Some(a) = bar() {
                    let b = 62;
                    1 + <|>
                }
            }
            ", r#"[CompletionItem { name: "b" },
                   CompletionItem { name: "a" }]"#);
    }

    #[test]
    fn test_completion_for_scope() {
        do_check(r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ", r#"[CompletionItem { name: "x" }]"#);
    }
}
