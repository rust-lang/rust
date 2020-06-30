use std::fmt;

use hir::{AsAssocItem, Attrs, HirFileId, InFile, Semantics};
use itertools::Itertools;
use ra_cfg::CfgExpr;
use ra_ide_db::RootDatabase;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner, DocCommentsOwner, ModuleItemOwner, NameOwner},
    match_ast, SyntaxNode,
};

use crate::{display::ToNav, FileId, NavigationTarget};

#[derive(Debug, Clone)]
pub struct Runnable {
    pub nav: NavigationTarget,
    pub kind: RunnableKind,
    pub cfg_exprs: Vec<CfgExpr>,
}

#[derive(Debug, Clone)]
pub enum TestId {
    Name(String),
    Path(String),
}

impl fmt::Display for TestId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TestId::Name(name) => write!(f, "{}", name),
            TestId::Path(path) => write!(f, "{}", path),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RunnableKind {
    Test { test_id: TestId, attr: TestAttr },
    TestMod { path: String },
    Bench { test_id: TestId },
    DocTest { test_id: TestId },
    Bin,
}

#[derive(Debug, Eq, PartialEq)]
pub struct RunnableAction {
    pub run_title: &'static str,
    pub debugee: bool,
}

const TEST: RunnableAction = RunnableAction { run_title: "▶\u{fe0e} Run Test", debugee: true };
const DOCTEST: RunnableAction =
    RunnableAction { run_title: "▶\u{fe0e} Run Doctest", debugee: false };
const BENCH: RunnableAction = RunnableAction { run_title: "▶\u{fe0e} Run Bench", debugee: true };
const BIN: RunnableAction = RunnableAction { run_title: "▶\u{fe0e} Run", debugee: true };

impl Runnable {
    // test package::module::testname
    pub fn label(&self, target: Option<String>) -> String {
        match &self.kind {
            RunnableKind::Test { test_id, .. } => format!("test {}", test_id),
            RunnableKind::TestMod { path } => format!("test-mod {}", path),
            RunnableKind::Bench { test_id } => format!("bench {}", test_id),
            RunnableKind::DocTest { test_id, .. } => format!("doctest {}", test_id),
            RunnableKind::Bin => {
                target.map_or_else(|| "run binary".to_string(), |t| format!("run {}", t))
            }
        }
    }

    pub fn action(&self) -> &'static RunnableAction {
        match &self.kind {
            RunnableKind::Test { .. } | RunnableKind::TestMod { .. } => &TEST,
            RunnableKind::DocTest { .. } => &DOCTEST,
            RunnableKind::Bench { .. } => &BENCH,
            RunnableKind::Bin => &BIN,
        }
    }
}

// Feature: Run
//
// Shows a popup suggesting to run a test/benchmark/binary **at the current cursor
// location**. Super useful for repeatedly running just a single test. Do bind this
// to a shortcut!
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Run**
// |===
pub(crate) fn runnables(db: &RootDatabase, file_id: FileId) -> Vec<Runnable> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(file_id);
    source_file.syntax().descendants().filter_map(|i| runnable(&sema, i, file_id)).collect()
}

pub(crate) fn runnable(
    sema: &Semantics<RootDatabase>,
    item: SyntaxNode,
    file_id: FileId,
) -> Option<Runnable> {
    match_ast! {
        match item {
            ast::FnDef(it) => runnable_fn(sema, it, file_id),
            ast::Module(it) => runnable_mod(sema, it, file_id),
            _ => None,
        }
    }
}

fn runnable_fn(
    sema: &Semantics<RootDatabase>,
    fn_def: ast::FnDef,
    file_id: FileId,
) -> Option<Runnable> {
    let name_string = fn_def.name()?.text().to_string();

    let kind = if name_string == "main" {
        RunnableKind::Bin
    } else {
        let test_id = match sema.to_def(&fn_def).map(|def| def.module(sema.db)) {
            Some(module) => {
                let def = sema.to_def(&fn_def)?;
                let impl_trait_name = def.as_assoc_item(sema.db).and_then(|assoc_item| {
                    match assoc_item.container(sema.db) {
                        hir::AssocItemContainer::Trait(trait_item) => {
                            Some(trait_item.name(sema.db).to_string())
                        }
                        hir::AssocItemContainer::ImplDef(impl_def) => impl_def
                            .target_ty(sema.db)
                            .as_adt()
                            .map(|adt| adt.name(sema.db).to_string()),
                    }
                });

                let path_iter = module
                    .path_to_root(sema.db)
                    .into_iter()
                    .rev()
                    .filter_map(|it| it.name(sema.db))
                    .map(|name| name.to_string());

                let path = if let Some(impl_trait_name) = impl_trait_name {
                    path_iter
                        .chain(std::iter::once(impl_trait_name))
                        .chain(std::iter::once(name_string))
                        .join("::")
                } else {
                    path_iter.chain(std::iter::once(name_string)).join("::")
                };

                TestId::Path(path)
            }
            None => TestId::Name(name_string),
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

    let attrs = Attrs::from_attrs_owner(sema.db, InFile::new(HirFileId::from(file_id), &fn_def));
    let cfg_exprs =
        attrs.by_key("cfg").tt_values().map(|subtree| ra_cfg::parse_cfg(subtree)).collect();

    let nav = if let RunnableKind::DocTest { .. } = kind {
        NavigationTarget::from_doc_commented(
            sema.db,
            InFile::new(file_id.into(), &fn_def),
            InFile::new(file_id.into(), &fn_def),
        )
    } else {
        NavigationTarget::from_named(sema.db, InFile::new(file_id.into(), &fn_def))
    };
    Some(Runnable { nav, kind, cfg_exprs })
}

#[derive(Debug, Copy, Clone)]
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

fn runnable_mod(
    sema: &Semantics<RootDatabase>,
    module: ast::Module,
    file_id: FileId,
) -> Option<Runnable> {
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
    let module_def = sema.to_def(&module)?;

    let path = module_def
        .path_to_root(sema.db)
        .into_iter()
        .rev()
        .filter_map(|it| it.name(sema.db))
        .join("::");

    let attrs = Attrs::from_attrs_owner(sema.db, InFile::new(HirFileId::from(file_id), &module));
    let cfg_exprs =
        attrs.by_key("cfg").tt_values().map(|subtree| ra_cfg::parse_cfg(subtree)).collect();

    let nav = module_def.to_nav(sema.db);
    Some(Runnable { nav, kind: RunnableKind::TestMod { path }, cfg_exprs })
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use crate::mock_analysis::analysis_and_position;

    use super::{RunnableAction, BENCH, BIN, DOCTEST, TEST};

    fn check(
        ra_fixture: &str,
        // FIXME: fold this into `expect` as well
        actions: &[&RunnableAction],
        expect: Expect,
    ) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        let runnables = analysis.runnables(position.file_id).unwrap();
        expect.assert_debug_eq(&runnables);
        assert_eq!(
            actions,
            runnables.into_iter().map(|it| it.action()).collect::<Vec<_>>().as_slice()
        );
    }

    #[test]
    fn test_runnables() {
        check(
            r#"
//- /lib.rs
<|>
fn main() {}

#[test]
fn test_foo() {}

#[test]
#[ignore]
fn test_foo() {}

#[bench]
fn bench() {}
"#,
            &[&BIN, &TEST, &TEST, &BENCH],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..13,
                            name: "main",
                            kind: FN_DEF,
                            focus_range: Some(
                                4..8,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Bin,
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 15..39,
                            name: "test_foo",
                            kind: FN_DEF,
                            focus_range: Some(
                                26..34,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 41..75,
                            name: "test_foo",
                            kind: FN_DEF,
                            focus_range: Some(
                                62..70,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo",
                            ),
                            attr: TestAttr {
                                ignore: true,
                            },
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 77..99,
                            name: "bench",
                            kind: FN_DEF,
                            focus_range: Some(
                                89..94,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Bench {
                            test_id: Path(
                                "bench",
                            ),
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test() {
        check(
            r#"
//- /lib.rs
<|>
fn main() {}

/// ```
/// let x = 5;
/// ```
fn foo() {}
"#,
            &[&BIN, &DOCTEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..13,
                            name: "main",
                            kind: FN_DEF,
                            focus_range: Some(
                                4..8,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Bin,
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 15..57,
                            name: "foo",
                            kind: FN_DEF,
                            focus_range: None,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: DocTest {
                            test_id: Path(
                                "foo",
                            ),
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl() {
        check(
            r#"
//- /lib.rs
<|>
fn main() {}

struct Data;
impl Data {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
"#,
            &[&BIN, &DOCTEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..13,
                            name: "main",
                            kind: FN_DEF,
                            focus_range: Some(
                                4..8,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Bin,
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 44..98,
                            name: "foo",
                            kind: FN_DEF,
                            focus_range: None,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: DocTest {
                            test_id: Path(
                                "Data::foo",
                            ),
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_module() {
        check(
            r#"
//- /lib.rs
<|>
mod test_mod {
    #[test]
    fn test_foo1() {}
}
"#,
            &[&TEST, &TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..51,
                            name: "test_mod",
                            kind: MODULE,
                            focus_range: Some(
                                5..13,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "test_mod",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 20..49,
                            name: "test_foo1",
                            kind: FN_DEF,
                            focus_range: Some(
                                35..44,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_mod::test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_one_depth_layer_module() {
        check(
            r#"
//- /lib.rs
<|>
mod foo {
    mod test_mod {
        #[test]
        fn test_foo1() {}
    }
}
"#,
            &[&TEST, &TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 15..77,
                            name: "test_mod",
                            kind: MODULE,
                            focus_range: Some(
                                19..27,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "foo::test_mod",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 38..71,
                            name: "test_foo1",
                            kind: FN_DEF,
                            focus_range: Some(
                                57..66,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "foo::test_mod::test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_multiple_depth_module() {
        check(
            r#"
//- /lib.rs
<|>
mod foo {
    mod bar {
        mod test_mod {
            #[test]
            fn test_foo1() {}
        }
    }
}
"#,
            &[&TEST, &TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 33..107,
                            name: "test_mod",
                            kind: MODULE,
                            focus_range: Some(
                                37..45,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "foo::bar::test_mod",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 60..97,
                            name: "test_foo1",
                            kind: FN_DEF,
                            focus_range: Some(
                                83..92,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "foo::bar::test_mod::test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_with_feature() {
        check(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
<|>
#[test]
#[cfg(feature = "foo")]
fn test_foo1() {}
"#,
            &[&TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..50,
                            name: "test_foo1",
                            kind: FN_DEF,
                            focus_range: Some(
                                36..45,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [
                            KeyValue {
                                key: "feature",
                                value: "foo",
                            },
                        ],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_with_features() {
        check(
            r#"
//- /lib.rs crate:foo cfg:feature=foo,feature=bar
<|>
#[test]
#[cfg(all(feature = "foo", feature = "bar"))]
fn test_foo1() {}
"#,
            &[&TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..72,
                            name: "test_foo1",
                            kind: FN_DEF,
                            focus_range: Some(
                                58..67,
                            ),
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg_exprs: [
                            All(
                                [
                                    KeyValue {
                                        key: "feature",
                                        value: "foo",
                                    },
                                    KeyValue {
                                        key: "feature",
                                        value: "bar",
                                    },
                                ],
                            ),
                        ],
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_no_test_function_in_module() {
        check(
            r#"
//- /lib.rs
<|>
mod test_mod {
    fn foo1() {}
}
"#,
            &[],
            expect![[r#"
                []
            "#]],
        );
    }
}
