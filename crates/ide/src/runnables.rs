use std::fmt;

use ast::NameOwner;
use cfg::CfgExpr;
use hir::{AsAssocItem, HasAttrs, HasSource, Semantics};
use ide_assists::utils::test_related_attribute;
use ide_db::{
    base_db::{FilePosition, FileRange},
    defs::Definition,
    search::SearchScope,
    RootDatabase, SymbolKind,
};
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, AstNode, AttrsOwner},
    match_ast, SyntaxNode,
};
use test_utils::mark;

use crate::{
    display::{ToNav, TryToNav},
    references, FileId, NavigationTarget,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Runnable {
    pub nav: NavigationTarget,
    pub kind: RunnableKind,
    pub cfg: Option<CfgExpr>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
    let module = match sema.to_module_def(file_id) {
        None => return Vec::new(),
        Some(it) => it,
    };

    let mut res = Vec::new();
    runnables_mod(&sema, &mut res, module);
    res
}

// Feature: Related Tests
//
// Provides a sneak peek of all tests where the current item is used.
//
// The simplest way to use this feature is via the context menu:
//  - Right-click on the selected item. The context menu opens.
//  - Select **Peek related tests**
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Peek related tests**
// |===
pub(crate) fn related_tests(
    db: &RootDatabase,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Vec<Runnable> {
    let sema = Semantics::new(db);
    let mut res: FxHashSet<Runnable> = FxHashSet::default();

    find_related_tests(&sema, position, search_scope, &mut res);

    res.into_iter().collect_vec()
}

fn find_related_tests(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    search_scope: Option<SearchScope>,
    tests: &mut FxHashSet<Runnable>,
) {
    if let Some(refs) = references::find_all_refs(&sema, position, search_scope) {
        for (file_id, refs) in refs.references {
            let file = sema.parse(file_id);
            let file = file.syntax();
            let functions = refs.iter().filter_map(|(range, _)| {
                let token = file.token_at_offset(range.start()).next()?;
                let token = sema.descend_into_macros(token);
                let syntax = token.parent();
                syntax.ancestors().find_map(ast::Fn::cast)
            });

            for fn_def in functions {
                if let Some(runnable) = as_test_runnable(&sema, &fn_def) {
                    // direct test
                    tests.insert(runnable);
                } else if let Some(module) = parent_test_module(&sema, &fn_def) {
                    // indirect test
                    find_related_tests_in_module(sema, &fn_def, &module, tests);
                }
            }
        }
    }
}

fn find_related_tests_in_module(
    sema: &Semantics<RootDatabase>,
    fn_def: &ast::Fn,
    parent_module: &hir::Module,
    tests: &mut FxHashSet<Runnable>,
) {
    if let Some(fn_name) = fn_def.name() {
        let mod_source = parent_module.definition_source(sema.db);
        let range = match mod_source.value {
            hir::ModuleSource::Module(m) => m.syntax().text_range(),
            hir::ModuleSource::BlockExpr(b) => b.syntax().text_range(),
            hir::ModuleSource::SourceFile(f) => f.syntax().text_range(),
        };

        let file_id = mod_source.file_id.original_file(sema.db);
        let mod_scope = SearchScope::file_range(FileRange { file_id, range });
        let fn_pos = FilePosition { file_id, offset: fn_name.syntax().text_range().start() };
        find_related_tests(sema, fn_pos, Some(mod_scope), tests)
    }
}

fn as_test_runnable(sema: &Semantics<RootDatabase>, fn_def: &ast::Fn) -> Option<Runnable> {
    if test_related_attribute(&fn_def).is_some() {
        let function = sema.to_def(fn_def)?;
        runnable_fn(sema, function)
    } else {
        None
    }
}

fn parent_test_module(sema: &Semantics<RootDatabase>, fn_def: &ast::Fn) -> Option<hir::Module> {
    fn_def.syntax().ancestors().find_map(|node| {
        let module = ast::Module::cast(node)?;
        let module = sema.to_def(&module)?;

        if has_test_function_or_multiple_test_submodules(sema, &module) {
            Some(module)
        } else {
            None
        }
    })
}

fn runnables_mod(sema: &Semantics<RootDatabase>, acc: &mut Vec<Runnable>, module: hir::Module) {
    acc.extend(module.declarations(sema.db).into_iter().filter_map(|def| {
        let runnable = match def {
            hir::ModuleDef::Module(it) => runnable_mod(&sema, it),
            hir::ModuleDef::Function(it) => runnable_fn(&sema, it),
            _ => None,
        };
        runnable.or_else(|| module_def_doctest(&sema, def))
    }));

    acc.extend(module.impl_defs(sema.db).into_iter().flat_map(|it| it.items(sema.db)).filter_map(
        |def| match def {
            hir::AssocItem::Function(it) => {
                runnable_fn(&sema, it).or_else(|| module_def_doctest(&sema, it.into()))
            }
            hir::AssocItem::Const(it) => module_def_doctest(&sema, it.into()),
            hir::AssocItem::TypeAlias(it) => module_def_doctest(&sema, it.into()),
        },
    ));

    for def in module.declarations(sema.db) {
        if let hir::ModuleDef::Module(submodule) = def {
            match submodule.definition_source(sema.db).value {
                hir::ModuleSource::Module(_) => runnables_mod(sema, acc, submodule),
                hir::ModuleSource::SourceFile(_) => mark::hit!(dont_recurse_in_outline_submodules),
                hir::ModuleSource::BlockExpr(_) => {} // inner items aren't runnable
            }
        }
    }
}

pub(crate) fn runnable_fn(sema: &Semantics<RootDatabase>, def: hir::Function) -> Option<Runnable> {
    let func = def.source(sema.db)?;
    let name_string = def.name(sema.db).to_string();

    let kind = if name_string == "main" {
        RunnableKind::Bin
    } else {
        let canonical_path = {
            let def: hir::ModuleDef = def.into();
            def.canonical_path(sema.db)
        };
        let test_id = canonical_path.map(TestId::Path).unwrap_or(TestId::Name(name_string));

        if test_related_attribute(&func.value).is_some() {
            let attr = TestAttr::from_fn(&func.value);
            RunnableKind::Test { test_id, attr }
        } else if func.value.has_atom_attr("bench") {
            RunnableKind::Bench { test_id }
        } else {
            return None;
        }
    };

    let nav = NavigationTarget::from_named(
        sema.db,
        func.as_ref().map(|it| it as &dyn ast::NameOwner),
        SymbolKind::Function,
    );
    let cfg = def.attrs(sema.db).cfg();
    Some(Runnable { nav, kind, cfg })
}

pub(crate) fn runnable_mod(sema: &Semantics<RootDatabase>, def: hir::Module) -> Option<Runnable> {
    if !has_test_function_or_multiple_test_submodules(sema, &def) {
        return None;
    }
    let path =
        def.path_to_root(sema.db).into_iter().rev().filter_map(|it| it.name(sema.db)).join("::");

    let attrs = def.attrs(sema.db);
    let cfg = attrs.cfg();
    let nav = def.to_nav(sema.db);
    Some(Runnable { nav, kind: RunnableKind::TestMod { path }, cfg })
}

// FIXME: figure out a proper API here.
pub(crate) fn doc_owner_to_def(
    sema: &Semantics<RootDatabase>,
    item: SyntaxNode,
) -> Option<Definition> {
    let res: hir::ModuleDef = match_ast! {
        match item {
            ast::SourceFile(it) => sema.scope(&item).module()?.into(),
            ast::Fn(it) => sema.to_def(&it)?.into(),
            ast::Struct(it) => sema.to_def(&it)?.into(),
            ast::Enum(it) => sema.to_def(&it)?.into(),
            ast::Union(it) => sema.to_def(&it)?.into(),
            ast::Trait(it) => sema.to_def(&it)?.into(),
            ast::Const(it) => sema.to_def(&it)?.into(),
            ast::Static(it) => sema.to_def(&it)?.into(),
            ast::TypeAlias(it) => sema.to_def(&it)?.into(),
            _ => return None,
        }
    };
    Some(Definition::ModuleDef(res))
}

fn module_def_doctest(sema: &Semantics<RootDatabase>, def: hir::ModuleDef) -> Option<Runnable> {
    let attrs = match def {
        hir::ModuleDef::Module(it) => it.attrs(sema.db),
        hir::ModuleDef::Function(it) => it.attrs(sema.db),
        hir::ModuleDef::Adt(it) => it.attrs(sema.db),
        hir::ModuleDef::Variant(it) => it.attrs(sema.db),
        hir::ModuleDef::Const(it) => it.attrs(sema.db),
        hir::ModuleDef::Static(it) => it.attrs(sema.db),
        hir::ModuleDef::Trait(it) => it.attrs(sema.db),
        hir::ModuleDef::TypeAlias(it) => it.attrs(sema.db),
        hir::ModuleDef::BuiltinType(_) => return None,
    };
    if !has_runnable_doc_test(&attrs) {
        return None;
    }
    let def_name = def.name(sema.db).map(|it| it.to_string());
    let test_id = def
        .canonical_path(sema.db)
        // This probably belongs to canonical path?
        .map(|path| {
            let assoc_def = match def {
                hir::ModuleDef::Function(it) => it.as_assoc_item(sema.db),
                hir::ModuleDef::Const(it) => it.as_assoc_item(sema.db),
                hir::ModuleDef::TypeAlias(it) => it.as_assoc_item(sema.db),
                _ => None,
            };
            // FIXME: this also looks very wrong
            if let Some(assoc_def) = assoc_def {
                if let hir::AssocItemContainer::Impl(imp) = assoc_def.container(sema.db) {
                    if let Some(adt) = imp.target_ty(sema.db).as_adt() {
                        let name = adt.name(sema.db).to_string();
                        let idx = path.rfind(':').map_or(0, |idx| idx + 1);
                        let (prefix, suffix) = path.split_at(idx);
                        return format!("{}{}::{}", prefix, name, suffix);
                    }
                }
            }
            path
        })
        .map(TestId::Path)
        .or_else(|| def_name.clone().map(TestId::Name))?;

    let mut nav = def.try_to_nav(sema.db)?;
    nav.focus_range = None;
    nav.description = None;
    nav.docs = None;
    nav.kind = None;
    let res = Runnable { nav, kind: RunnableKind::DocTest { test_id }, cfg: attrs.cfg() };
    Some(res)
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
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

const RUSTDOC_FENCE: &str = "```";
const RUSTDOC_CODE_BLOCK_ATTRIBUTES_RUNNABLE: &[&str] =
    &["", "rust", "should_panic", "edition2015", "edition2018", "edition2021"];

fn has_runnable_doc_test(attrs: &hir::Attrs) -> bool {
    attrs.docs().map_or(false, |doc| {
        let mut in_code_block = false;

        for line in String::from(doc).lines() {
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

// We could create runnables for modules with number_of_test_submodules > 0,
// but that bloats the runnables for no real benefit, since all tests can be run by the submodule already
fn has_test_function_or_multiple_test_submodules(
    sema: &Semantics<RootDatabase>,
    module: &hir::Module,
) -> bool {
    let mut number_of_test_submodules = 0;

    for item in module.declarations(sema.db) {
        match item {
            hir::ModuleDef::Function(f) => {
                if let Some(it) = f.source(sema.db) {
                    if test_related_attribute(&it.value).is_some() {
                        return true;
                    }
                }
            }
            hir::ModuleDef::Module(submodule) => {
                if has_test_function_or_multiple_test_submodules(sema, &submodule) {
                    number_of_test_submodules += 1;
                }
            }
            _ => (),
        }
    }

    number_of_test_submodules > 1
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::fixture;

    use super::*;

    fn check(
        ra_fixture: &str,
        // FIXME: fold this into `expect` as well
        actions: &[&RunnableAction],
        expect: Expect,
    ) {
        let (analysis, position) = fixture::position(ra_fixture);
        let runnables = analysis.runnables(position.file_id).unwrap();
        expect.assert_debug_eq(&runnables);
        assert_eq!(
            actions,
            runnables.into_iter().map(|it| it.action()).collect::<Vec<_>>().as_slice()
        );
    }

    fn check_tests(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let tests = analysis.related_tests(position, None).unwrap();
        expect.assert_debug_eq(&tests);
    }

    #[test]
    fn test_runnables() {
        check(
            r#"
//- /lib.rs
$0
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
                                0,
                            ),
                            full_range: 1..13,
                            focus_range: 4..8,
                            name: "main",
                            kind: Function,
                        },
                        kind: Bin,
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 15..39,
                            focus_range: 26..34,
                            name: "test_foo",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 41..75,
                            focus_range: 62..70,
                            name: "test_foo",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo",
                            ),
                            attr: TestAttr {
                                ignore: true,
                            },
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 77..99,
                            focus_range: 89..94,
                            name: "bench",
                            kind: Function,
                        },
                        kind: Bench {
                            test_id: Path(
                                "bench",
                            ),
                        },
                        cfg: None,
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
$0
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

/// ```
/// let x = 5;
/// ```
struct StructWithRunnable(String);

"#,
            &[&BIN, &DOCTEST, &DOCTEST, &DOCTEST, &DOCTEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 1..13,
                            focus_range: 4..8,
                            name: "main",
                            kind: Function,
                        },
                        kind: Bin,
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 15..74,
                            name: "should_have_runnable",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable",
                            ),
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 76..148,
                            name: "should_have_runnable_1",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable_1",
                            ),
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 150..254,
                            name: "should_have_runnable_2",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "should_have_runnable_2",
                            ),
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 756..821,
                            name: "StructWithRunnable",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "StructWithRunnable",
                            ),
                        },
                        cfg: None,
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
$0
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
                                0,
                            ),
                            full_range: 1..13,
                            focus_range: 4..8,
                            name: "main",
                            kind: Function,
                        },
                        kind: Bin,
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 44..98,
                            name: "foo",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "Data::foo",
                            ),
                        },
                        cfg: None,
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
$0
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
                                0,
                            ),
                            full_range: 1..51,
                            focus_range: 5..13,
                            name: "test_mod",
                            kind: Module,
                        },
                        kind: TestMod {
                            path: "test_mod",
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 20..49,
                            focus_range: 35..44,
                            name: "test_foo1",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_mod::test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
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
$0
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
                                0,
                            ),
                            full_range: 22..323,
                            focus_range: 26..40,
                            name: "nested_tests_0",
                            kind: Module,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0",
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 51..192,
                            focus_range: 55..69,
                            name: "nested_tests_1",
                            kind: Module,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0::nested_tests_1",
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 202..286,
                            focus_range: 206..220,
                            name: "nested_tests_2",
                            kind: Module,
                        },
                        kind: TestMod {
                            path: "root_tests::nested_tests_0::nested_tests_2",
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 84..126,
                            focus_range: 107..121,
                            name: "nested_test_11",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_1::nested_test_11",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 140..182,
                            focus_range: 163..177,
                            name: "nested_test_12",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_1::nested_test_12",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 235..276,
                            focus_range: 258..271,
                            name: "nested_test_2",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "root_tests::nested_tests_0::nested_tests_2::nested_test_2",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
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
$0
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
                                0,
                            ),
                            full_range: 1..50,
                            focus_range: 36..45,
                            name: "test_foo1",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: Some(
                            Atom(
                                KeyValue {
                                    key: "feature",
                                    value: "foo",
                                },
                            ),
                        ),
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
$0
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
                                0,
                            ),
                            full_range: 1..72,
                            focus_range: 58..67,
                            name: "test_foo1",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test_foo1",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: Some(
                            All(
                                [
                                    Atom(
                                        KeyValue {
                                            key: "feature",
                                            value: "foo",
                                        },
                                    ),
                                    Atom(
                                        KeyValue {
                                            key: "feature",
                                            value: "bar",
                                        },
                                    ),
                                ],
                            ),
                        ),
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
$0
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

    #[test]
    fn test_doc_runnables_impl_mod() {
        check(
            r#"
//- /lib.rs
mod foo;
//- /foo.rs
struct Foo;$0
impl Foo {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
        "#,
            &[&DOCTEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                1,
                            ),
                            full_range: 27..81,
                            name: "foo",
                        },
                        kind: DocTest {
                            test_id: Path(
                                "foo::Foo::foo",
                            ),
                        },
                        cfg: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_in_macro() {
        check(
            r#"
//- /lib.rs
$0
macro_rules! gen {
    () => {
        #[test]
        fn foo_test() {
        }
    }
}
mod tests {
    gen!();
}
"#,
            &[&TEST, &TEST],
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 90..115,
                            focus_range: 94..99,
                            name: "tests",
                            kind: Module,
                        },
                        kind: TestMod {
                            path: "tests",
                        },
                        cfg: None,
                    },
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 106..113,
                            focus_range: 106..113,
                            name: "foo_test",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "tests::foo_test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn dont_recurse_in_outline_submodules() {
        mark::check!(dont_recurse_in_outline_submodules);
        check(
            r#"
//- /lib.rs
$0
mod m;
//- /m.rs
mod tests {
    #[test]
    fn t() {}
}
"#,
            &[],
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
    fn find_no_tests() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() {  };
"#,
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
    fn find_direct_fn_test() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() { };

mod tests {
    #[test]
    fn foo_test() {
        super::foo()
    }
}
"#,
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 31..85,
                            focus_range: 46..54,
                            name: "foo_test",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "tests::foo_test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn find_direct_struct_test() {
        check_tests(
            r#"
//- /lib.rs
struct Fo$0o;
fn foo(arg: &Foo) { };

mod tests {
    use super::*;

    #[test]
    fn foo_test() {
        foo(Foo);
    }
}
"#,
            expect![[r#"
            [
                Runnable {
                    nav: NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 71..122,
                        focus_range: 86..94,
                        name: "foo_test",
                        kind: Function,
                    },
                    kind: Test {
                        test_id: Path(
                            "tests::foo_test",
                        ),
                        attr: TestAttr {
                            ignore: false,
                        },
                    },
                    cfg: None,
                },
            ]
            "#]],
        );
    }

    #[test]
    fn find_indirect_fn_test() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() { };

mod tests {
    use super::foo;

    fn check1() {
        check2()
    }

    fn check2() {
        foo()
    }

    #[test]
    fn foo_test() {
        check1()
    }
}
"#,
            expect![[r#"
                [
                    Runnable {
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 133..183,
                            focus_range: 148..156,
                            name: "foo_test",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "tests::foo_test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn tests_are_unique() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() { };

mod tests {
    use super::foo;

    #[test]
    fn foo_test() {
        foo();
        foo();
    }

    #[test]
    fn foo2_test() {
        foo();
        foo();
    }

}
"#,
            expect![[r#"
            [
                Runnable {
                    nav: NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 52..115,
                        focus_range: 67..75,
                        name: "foo_test",
                        kind: Function,
                    },
                    kind: Test {
                        test_id: Path(
                            "tests::foo_test",
                        ),
                        attr: TestAttr {
                            ignore: false,
                        },
                    },
                    cfg: None,
                },
                Runnable {
                    nav: NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 121..185,
                        focus_range: 136..145,
                        name: "foo2_test",
                        kind: Function,
                    },
                    kind: Test {
                        test_id: Path(
                            "tests::foo2_test",
                        ),
                        attr: TestAttr {
                            ignore: false,
                        },
                    },
                    cfg: None,
                },
            ]
            "#]],
        );
    }
}
