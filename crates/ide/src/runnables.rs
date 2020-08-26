use std::fmt;

use cfg::CfgExpr;
use hir::{AsAssocItem, Attrs, HirFileId, InFile, Semantics};
use ide_db::RootDatabase;
use itertools::Itertools;
use syntax::{
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
            ast::Fn(it) => runnable_fn(sema, it, file_id),
            ast::Module(it) => runnable_mod(sema, it, file_id),
            _ => None,
        }
    }
}

fn runnable_fn(
    sema: &Semantics<RootDatabase>,
    fn_def: ast::Fn,
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
        } else if has_runnable_doc_test(&fn_def) {
            RunnableKind::DocTest { test_id }
        } else {
            return None;
        }
    };

    let attrs = Attrs::from_attrs_owner(sema.db, InFile::new(HirFileId::from(file_id), &fn_def));
    let cfg_exprs = attrs.cfg().collect();

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
    fn from_fn(fn_def: &ast::Fn) -> TestAttr {
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
fn has_test_related_attribute(fn_def: &ast::Fn) -> bool {
    fn_def
        .attrs()
        .filter_map(|attr| attr.path())
        .map(|path| path.syntax().to_string().to_lowercase())
        .any(|attribute_text| attribute_text.contains("test"))
}

const RUSTDOC_FENCE: &str = "```";
const RUSTDOC_CODE_BLOCK_ATTRIBUTES_RUNNABLE: &[&str] =
    &["", "rust", "should_panic", "edition2015", "edition2018"];

fn has_runnable_doc_test(fn_def: &ast::Fn) -> bool {
    fn_def.doc_comment_text().map_or(false, |comments_text| {
        let mut in_code_block = false;

        for line in comments_text.lines() {
            if let Some(header) = line.strip_prefix(RUSTDOC_FENCE) {
                in_code_block = !in_code_block;

                if in_code_block
                    && header
                        .split(',')
                        .all(|sub| RUSTDOC_CODE_BLOCK_ATTRIBUTES_RUNNABLE.contains(&sub.trim()))
                {
                    return true;
                }
            }
        }

        false
    })
}

fn runnable_mod(
    sema: &Semantics<RootDatabase>,
    module: ast::Module,
    file_id: FileId,
) -> Option<Runnable> {
    if !has_test_function_or_multiple_test_submodules(&module) {
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
    let cfg_exprs = attrs.cfg().collect();
    let nav = module_def.to_nav(sema.db);
    Some(Runnable { nav, kind: RunnableKind::TestMod { path }, cfg_exprs })
}

// We could create runnables for modules with number_of_test_submodules > 0,
// but that bloats the runnables for no real benefit, since all tests can be run by the submodule already
fn has_test_function_or_multiple_test_submodules(module: &ast::Module) -> bool {
    if let Some(item_list) = module.item_list() {
        let mut number_of_test_submodules = 0;

        for item in item_list.items() {
            match item {
                ast::Item::Fn(f) => {
                    if has_test_related_attribute(&f) {
                        return true;
                    }
                }
                ast::Item::Module(submodule) => {
                    if has_test_function_or_multiple_test_submodules(&submodule) {
                        number_of_test_submodules += 1;
                    }
                }
                _ => (),
            }
        }

        number_of_test_submodules > 1
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

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
                            focus_range: Some(
                                4..8,
                            ),
                            name: "main",
                            kind: FN,
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
                            focus_range: Some(
                                26..34,
                            ),
                            name: "test_foo",
                            kind: FN,
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
                            focus_range: Some(
                                62..70,
                            ),
                            name: "test_foo",
                            kind: FN,
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
                            focus_range: Some(
                                89..94,
                            ),
                            name: "bench",
                            kind: FN,
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
fn should_have_runnable() {}

/// ```edition2018
/// let x = 5;
/// ```
fn should_have_runnable_1() {}

/// ```
/// let z = 55;
/// ```
///
/// ```ignore
/// let z = 56;
/// ```
fn should_have_runnable_2() {}

/// ```no_run
/// let z = 55;
/// ```
fn should_have_no_runnable() {}

/// ```ignore
/// let z = 55;
/// ```
fn should_have_no_runnable_2() {}

/// ```compile_fail
/// let z = 55;
/// ```
fn should_have_no_runnable_3() {}

/// ```text
/// arbitrary plain text
/// ```
fn should_have_no_runnable_4() {}

/// ```text
/// arbitrary plain text
/// ```
///
/// ```sh
/// $ shell code
/// ```
fn should_have_no_runnable_5() {}

/// ```rust,no_run
/// let z = 55;
/// ```
fn should_have_no_runnable_6() {}
"#,
            &[&BIN, &DOCTEST, &DOCTEST, &DOCTEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 1..13,
                            focus_range: Some(
                                4..8,
                            ),
                            name: "main",
                            kind: FN,
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
                            full_range: 15..74,
                            focus_range: None,
                            name: "should_have_runnable",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable",
                            ),
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 76..148,
                            focus_range: None,
                            name: "should_have_runnable_1",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable_1",
                            ),
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 150..254,
                            focus_range: None,
                            name: "should_have_runnable_2",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable_2",
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
                            focus_range: Some(
                                4..8,
                            ),
                            name: "main",
                            kind: FN,
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
                            focus_range: None,
                            name: "foo",
                            kind: FN,
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
                            focus_range: Some(
                                5..13,
                            ),
                            name: "test_mod",
                            kind: MODULE,
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
                            focus_range: Some(
                                35..44,
                            ),
                            name: "test_foo1",
                            kind: FN,
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
    fn only_modules_with_test_functions_or_more_than_one_test_submodule_have_runners() {
        check(
            r#"
//- /lib.rs
<|>
mod root_tests {
    mod nested_tests_0 {
        mod nested_tests_1 {
            #[test]
            fn nested_test_11() {}

            #[test]
            fn nested_test_12() {}
        }

        mod nested_tests_2 {
            #[test]
            fn nested_test_2() {}
        }

        mod nested_tests_3 {}
    }

    mod nested_tests_4 {}
}
"#,
            &[&TEST, &TEST, &TEST, &TEST, &TEST, &TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 22..323,
                            focus_range: Some(
                                26..40,
                            ),
                            name: "nested_tests_0",
                            kind: MODULE,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 51..192,
                            focus_range: Some(
                                55..69,
                            ),
                            name: "nested_tests_1",
                            kind: MODULE,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0::nested_tests_1",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 84..126,
                            focus_range: Some(
                                107..121,
                            ),
                            name: "nested_test_11",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_1::nested_test_11",
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
                            full_range: 140..182,
                            focus_range: Some(
                                163..177,
                            ),
                            name: "nested_test_12",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_1::nested_test_12",
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
                            full_range: 202..286,
                            focus_range: Some(
                                206..220,
                            ),
                            name: "nested_tests_2",
                            kind: MODULE,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0::nested_tests_2",
                        },
                        cfg_exprs: [],
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 235..276,
                            focus_range: Some(
                                258..271,
                            ),
                            name: "nested_test_2",
                            kind: FN,
                            container_name: None,
                            description: None,
                            docs: None,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_2::nested_test_2",
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
                            focus_range: Some(
                                36..45,
                            ),
                            name: "test_foo1",
                            kind: FN,
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
                            focus_range: Some(
                                58..67,
                            ),
                            name: "test_foo1",
                            kind: FN,
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
