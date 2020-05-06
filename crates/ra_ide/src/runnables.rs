//! FIXME: write short doc here

use hir::Semantics;
use itertools::Itertools;
use ra_ide_db::RootDatabase;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner, ModuleItemOwner, NameOwner},
    match_ast, SyntaxNode, TextRange,
};

use crate::FileId;
use ast::DocCommentsOwner;
use std::fmt::Display;

#[derive(Debug)]
pub struct Runnable {
    pub range: TextRange,
    pub kind: RunnableKind,
}

#[derive(Debug)]
pub enum TestId {
    Name(String),
    Path(String),
}

impl Display for TestId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TestId::Name(name) => write!(f, "{}", name),
            TestId::Path(path) => write!(f, "{}", path),
        }
    }
}

#[derive(Debug)]
pub enum RunnableKind {
    Test { test_id: TestId, attr: TestAttr },
    TestMod { path: String },
    Bench { test_id: TestId },
    DocTest { test_id: TestId },
    Bin,
}

pub(crate) fn runnables(db: &RootDatabase, file_id: FileId) -> Vec<Runnable> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(file_id);
    source_file.syntax().descendants().filter_map(|i| runnable(&sema, i)).collect()
}

fn runnable(sema: &Semantics<RootDatabase>, item: SyntaxNode) -> Option<Runnable> {
    match_ast! {
        match item {
            ast::FnDef(it) => runnable_fn(sema, it),
            ast::Module(it) => runnable_mod(sema, it),
            _ => None,
        }
    }
}

fn runnable_fn(sema: &Semantics<RootDatabase>, fn_def: ast::FnDef) -> Option<Runnable> {
    let name_string = fn_def.name()?.text().to_string();

    let kind = if name_string == "main" {
        RunnableKind::Bin
    } else {
        let test_id = if let Some(module) = sema.to_def(&fn_def).map(|def| def.module(sema.db)) {
            let path = module
                .path_to_root(sema.db)
                .into_iter()
                .rev()
                .filter_map(|it| it.name(sema.db))
                .map(|name| name.to_string())
                .chain(std::iter::once(name_string))
                .join("::");
            TestId::Path(path)
        } else {
            TestId::Name(name_string)
        };

        if has_test_related_attribute(&fn_def) {
            let attr = TestAttr::from_fn(&fn_def);
            RunnableKind::Test { test_id, attr }
        } else if fn_def.has_atom_attr("bench") {
            RunnableKind::Bench { test_id }
        } else if has_doc_test(&fn_def) {
            RunnableKind::DocTest { test_id }
        } else {
            return None;
        }
    };
    Some(Runnable { range: fn_def.syntax().text_range(), kind })
}

#[derive(Debug)]
pub struct TestAttr {
    pub ignore: bool,
}

impl TestAttr {
    fn from_fn(fn_def: &ast::FnDef) -> TestAttr {
        let ignore = fn_def
            .attrs()
            .filter_map(|attr| attr.simple_name())
            .any(|attribute_text| attribute_text == "ignore");
        TestAttr { ignore }
    }
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

fn has_doc_test(fn_def: &ast::FnDef) -> bool {
    fn_def.doc_comment_text().map_or(false, |comment| comment.contains("```"))
}

fn runnable_mod(sema: &Semantics<RootDatabase>, module: ast::Module) -> Option<Runnable> {
    let has_test_function = module
        .item_list()?
        .items()
        .filter_map(|it| match it {
            ast::ModuleItem::FnDef(it) => Some(it),
            _ => None,
        })
        .any(|f| has_test_related_attribute(&f));
    if !has_test_function {
        return None;
    }
    let range = module.syntax().text_range();
    let module = sema.to_def(&module)?;

    let path =
        module.path_to_root(sema.db).into_iter().rev().filter_map(|it| it.name(sema.db)).join("::");
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
                range: 1..21,
                kind: Bin,
            },
            Runnable {
                range: 22..46,
                kind: Test {
                    test_id: Path(
                        "test_foo",
                    ),
                    attr: TestAttr {
                        ignore: false,
                    },
                },
            },
            Runnable {
                range: 47..81,
                kind: Test {
                    test_id: Path(
                        "test_foo",
                    ),
                    attr: TestAttr {
                        ignore: true,
                    },
                },
            },
        ]
        "###
                );
    }

    #[test]
    fn test_runnables_doc_test() {
        let (analysis, pos) = analysis_and_position(
            r#"
        //- /lib.rs
        <|> //empty
        fn main() {}

        /// ```
        /// let x = 5;
        /// ```
        fn foo() {}
        "#,
        );
        let runnables = analysis.runnables(pos.file_id).unwrap();
        assert_debug_snapshot!(&runnables,
        @r###"
        [
            Runnable {
                range: 1..21,
                kind: Bin,
            },
            Runnable {
                range: 22..64,
                kind: DocTest {
                    test_id: Path(
                        "foo",
                    ),
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
                range: 1..59,
                kind: TestMod {
                    path: "test_mod",
                },
            },
            Runnable {
                range: 28..57,
                kind: Test {
                    test_id: Path(
                        "test_mod::test_foo1",
                    ),
                    attr: TestAttr {
                        ignore: false,
                    },
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
                range: 23..85,
                kind: TestMod {
                    path: "foo::test_mod",
                },
            },
            Runnable {
                range: 46..79,
                kind: Test {
                    test_id: Path(
                        "foo::test_mod::test_foo1",
                    ),
                    attr: TestAttr {
                        ignore: false,
                    },
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
                range: 41..115,
                kind: TestMod {
                    path: "foo::bar::test_mod",
                },
            },
            Runnable {
                range: 68..105,
                kind: Test {
                    test_id: Path(
                        "foo::bar::test_mod::test_foo1",
                    ),
                    attr: TestAttr {
                        ignore: false,
                    },
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
