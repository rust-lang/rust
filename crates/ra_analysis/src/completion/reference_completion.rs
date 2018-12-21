use rustc_hash::{FxHashSet};
use ra_syntax::{
    SourceFileNode, AstNode,
    ast,
    SyntaxKind::*,
};
use hir::{
    self,
    FnScopes, Def, Path
};

use crate::{
    db::RootDatabase,
    completion::{CompletionItem, Completions, CompletionKind::*},
    Cancelable
};

pub(super) fn completions(
    acc: &mut Completions,
    db: &RootDatabase,
    module: &hir::Module,
    _file: &SourceFileNode,
    name_ref: ast::NameRef,
) -> Cancelable<()> {
    let kind = match classify_name_ref(name_ref) {
        Some(it) => it,
        None => return Ok(()),
    };

    match kind {
        NameRefKind::LocalRef { enclosing_fn } => {
            if let Some(fn_def) = enclosing_fn {
                let scopes = FnScopes::new(fn_def);
                complete_fn(name_ref, &scopes, acc);
            }

            let module_scope = module.scope(db)?;
            module_scope
                .entries()
                .filter(|(_name, res)| {
                    // Don't expose this item
                    match res.import {
                        None => true,
                        Some(import) => {
                            let range = import.range(db, module.source().file_id());
                            !range.is_subrange(&name_ref.syntax().range())
                        }
                    }
                })
                .for_each(|(name, _res)| {
                    CompletionItem::new(name.to_string())
                        .kind(Reference)
                        .add_to(acc)
                });
        }
        NameRefKind::Path(path) => complete_path(acc, db, module, path)?,
        NameRefKind::BareIdentInMod => (),
    }
    Ok(())
}

enum NameRefKind<'a> {
    /// NameRef is a part of single-segment path, for example, a refernece to a
    /// local variable.
    LocalRef {
        enclosing_fn: Option<ast::FnDef<'a>>,
    },
    /// NameRef is the last segment in some path
    Path(Path),
    /// NameRef is bare identifier at the module's root.
    /// Used for keyword completion
    BareIdentInMod,
}

fn classify_name_ref(name_ref: ast::NameRef) -> Option<NameRefKind> {
    let name_range = name_ref.syntax().range();
    let top_node = name_ref
        .syntax()
        .ancestors()
        .take_while(|it| it.range() == name_range)
        .last()
        .unwrap();
    match top_node.parent().map(|it| it.kind()) {
        Some(SOURCE_FILE) | Some(ITEM_LIST) => return Some(NameRefKind::BareIdentInMod),
        _ => (),
    }

    let parent = name_ref.syntax().parent()?;
    if let Some(segment) = ast::PathSegment::cast(parent) {
        let path = segment.parent_path();
        if let Some(path) = Path::from_ast(path) {
            if !path.is_ident() {
                return Some(NameRefKind::Path(path));
            }
        }
        if path.qualifier().is_none() {
            let enclosing_fn = name_ref
                .syntax()
                .ancestors()
                .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
                .find_map(ast::FnDef::cast);
            return Some(NameRefKind::LocalRef { enclosing_fn });
        }
    }
    None
}

fn complete_fn(name_ref: ast::NameRef, scopes: &FnScopes, acc: &mut Completions) {
    let mut shadowed = FxHashSet::default();
    scopes
        .scope_chain(name_ref.syntax())
        .flat_map(|scope| scopes.entries(scope).iter())
        .filter(|entry| shadowed.insert(entry.name()))
        .for_each(|entry| {
            CompletionItem::new(entry.name().to_string())
                .kind(Reference)
                .add_to(acc)
        });
    if scopes.self_param.is_some() {
        CompletionItem::new("self").kind(Reference).add_to(acc);
    }
}

fn complete_path(
    acc: &mut Completions,
    db: &RootDatabase,
    module: &hir::Module,
    mut path: Path,
) -> Cancelable<()> {
    if path.segments.is_empty() {
        return Ok(());
    }
    path.segments.pop();
    let def_id = match module.resolve_path(db, path)? {
        None => return Ok(()),
        Some(it) => it,
    };
    let target_module = match def_id.resolve(db)? {
        Def::Module(it) => it,
        _ => return Ok(()),
    };
    let module_scope = target_module.scope(db)?;
    module_scope.entries().for_each(|(name, _res)| {
        CompletionItem::new(name.to_string())
            .kind(Reference)
            .add_to(acc)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn test_completion_let_scope() {
        check_reference_completion(
            r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ",
            "y;x;quux",
        );
    }

    #[test]
    fn test_completion_if_let_scope() {
        check_reference_completion(
            r"
            fn quux() {
                if let Some(x) = foo() {
                    let y = 92;
                };
                if let Some(a) = bar() {
                    let b = 62;
                    1 + <|>
                }
            }
            ",
            "b;a;quux",
        );
    }

    #[test]
    fn test_completion_for_scope() {
        check_reference_completion(
            r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ",
            "x;quux",
        );
    }

    #[test]
    fn test_completion_mod_scope() {
        check_reference_completion(
            r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ",
            "quux;Foo;Baz",
        );
    }

    #[test]
    fn test_completion_mod_scope_no_self_use() {
        check_reference_completion(
            r"
            use foo<|>;
            ",
            "",
        );
    }

    #[test]
    fn test_completion_self_path() {
        check_reference_completion(
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
            "Bar",
        );
    }

    #[test]
    fn test_completion_mod_scope_nested() {
        check_reference_completion(
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
            "quux;Bar",
        );
    }

    #[test]
    fn test_complete_type() {
        check_reference_completion(
            r"
            struct Foo;
            fn x() -> <|>
            ",
            "Foo;x",
        )
    }

    #[test]
    fn test_complete_shadowing() {
        check_reference_completion(
            r"
            fn foo() -> {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
            ",
            "bar;foo",
        )
    }

    #[test]
    fn test_complete_self() {
        check_reference_completion(r"impl S { fn foo(&self) { <|> } }", "self")
    }

    #[test]
    fn test_complete_crate_path() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::Sp<|>
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn test_complete_crate_path_with_braces() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::{Sp<|>};
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn test_complete_crate_path_in_nested_tree() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            pub mod bar {
                pub mod baz {
                    pub struct Spam;
                }
            }
            //- /foo.rs
            use crate::{bar::{baz::Sp<|>}};
            ",
            "Spam",
        );
    }
}
