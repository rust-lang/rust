//! FIXME: write short doc here

use hir::InFile;
use itertools::Itertools;
use ra_db::SourceDatabase;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner, ModuleItemOwner, NameOwner},
    match_ast, SyntaxNode, TextRange,
};

use crate::{db::RootDatabase, FileId};

#[derive(Debug)]
pub struct Runnable {
    pub range: TextRange,
    pub kind: RunnableKind,
}

#[derive(Debug)]
pub enum RunnableKind {
    Test { name: String },
    TestMod { path: String },
    Bench { name: String },
    Bin,
}

pub(crate) fn runnables(db: &RootDatabase, file_id: FileId) -> Vec<Runnable> {
    let parse = db.parse(file_id);
    parse.tree().syntax().descendants().filter_map(|i| runnable(db, file_id, i)).collect()
}

fn runnable(db: &RootDatabase, file_id: FileId, item: SyntaxNode) -> Option<Runnable> {
    match_ast! {
        match item {
            ast::FnDef(it) => { runnable_fn(it) },
            ast::Module(it) => { runnable_mod(db, file_id, it) },
            _ => { None },
        }
    }
}

fn runnable_fn(fn_def: ast::FnDef) -> Option<Runnable> {
    let name = fn_def.name()?.text().clone();
    let kind = if name == "main" {
        RunnableKind::Bin
    } else if has_test_related_attribute(&fn_def) {
        RunnableKind::Test { name: name.to_string() }
    } else if fn_def.has_atom_attr("bench") {
        RunnableKind::Bench { name: name.to_string() }
    } else {
        return None;
    };
    Some(Runnable { range: fn_def.syntax().text_range(), kind })
}

/// This is a method with a heuristics to support test methods annotated with custom test annotations, such as
/// `#[test_case(...)]`, `#[tokio::test]` and similar.
/// Also a regular `#[test]` annotation is supported.
///
/// It may produce false positives, for example, `#[wasm_bindgen_test]` requires a different command to run the test,
/// but it's better than not to have the runnables for the tests at all.
fn has_test_related_attribute(fn_def: &ast::FnDef) -> bool {
    fn_def
        .attrs()
        .filter_map(|attr| attr.path())
        .map(|path| path.syntax().to_string().to_lowercase())
        .any(|attribute_text| attribute_text.contains("test"))
}

fn runnable_mod(db: &RootDatabase, file_id: FileId, module: ast::Module) -> Option<Runnable> {
    let has_test_function = module
        .item_list()?
        .items()
        .filter_map(|it| match it {
            ast::ModuleItem::FnDef(it) => Some(it),
            _ => None,
        })
        .any(|f| f.has_atom_attr("test"));
    if !has_test_function {
        return None;
    }
    let range = module.syntax().text_range();
    let mut sb = hir::SourceBinder::new(db);
    let module = sb.to_def(InFile::new(file_id.into(), module))?;

    let path = module.path_to_root(db).into_iter().rev().filter_map(|it| it.name(db)).join("::");
    Some(Runnable { range, kind: RunnableKind::TestMod { path } })
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::mock_analysis::analysis_and_position;

    #[test]
    fn test_runnables() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        fn main() {}

        #[test]
        fn test_foo() {}

        #[test]
        #[ignore]
        fn test_foo() {}
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert_debug_snapshot!(&runnables,
        @r###"
        [
            Runnable {
                range: [1; 21),
                kind: Bin,
            },
            Runnable {
                range: [22; 46),
                kind: Test {
                    name: "test_foo",
                },
            },
            Runnable {
                range: [47; 81),
                kind: Test {
                    name: "test_foo",
                },
            },
        ]
        "###
                );
    }

    #[test]
    fn test_runnables_module() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        mod test_mod {
            #[test]
            fn test_foo1() {}
        }
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert_debug_snapshot!(&runnables,
        @r###"
        [
            Runnable {
                range: [1; 59),
                kind: TestMod {
                    path: "test_mod",
                },
            },
            Runnable {
                range: [28; 57),
                kind: Test {
                    name: "test_foo1",
                },
            },
        ]
        "###
                );
    }

    #[test]
    fn test_runnables_one_depth_layer_module() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        mod foo {
            mod test_mod {
                #[test]
                fn test_foo1() {}
            }
        }
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert_debug_snapshot!(&runnables,
        @r###"
        [
            Runnable {
                range: [23; 85),
                kind: TestMod {
                    path: "foo::test_mod",
                },
            },
            Runnable {
                range: [46; 79),
                kind: Test {
                    name: "test_foo1",
                },
            },
        ]
        "###
                );
    }

    #[test]
    fn test_runnables_multiple_depth_module() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        mod foo {
            mod bar {
                mod test_mod {
                    #[test]
                    fn test_foo1() {}
                }
            }
        }
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert_debug_snapshot!(&runnables,
        @r###"
        [
            Runnable {
                range: [41; 115),
                kind: TestMod {
                    path: "foo::bar::test_mod",
                },
            },
            Runnable {
                range: [68; 105),
                kind: Test {
                    name: "test_foo1",
                },
            },
        ]
        "###
                );
    }

    #[test]
    fn test_runnables_no_test_function_in_module() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        mod test_mod {
            fn foo1() {}
        }
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert!(runnables.is_empty())
    }
}
