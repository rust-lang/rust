use std::fmt;

use ast::HasName;
use cfg::CfgExpr;
use hir::{db::HirDatabase, AsAssocItem, HasAttrs, HasSource, HirFileIdExt, Semantics};
use ide_assists::utils::test_related_attribute;
use ide_db::{
    base_db::{FilePosition, FileRange},
    defs::Definition,
    documentation::docs_from_attrs,
    helpers::visit_file_defs,
    search::{FileReferenceNode, SearchScope},
    FxHashMap, FxHashSet, RootDatabase, SymbolKind,
};
use itertools::Itertools;
use stdx::{always, format_to};
use syntax::{
    ast::{self, AstNode},
    SmolStr, SyntaxNode,
};

use crate::{references, FileId, NavigationTarget, ToNav, TryToNav};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Runnable {
    pub use_name_in_title: bool,
    pub nav: NavigationTarget,
    pub kind: RunnableKind,
    pub cfg: Option<CfgExpr>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum TestId {
    Name(SmolStr),
    Path(String),
}

impl fmt::Display for TestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestId::Name(name) => name.fmt(f),
            TestId::Path(path) => path.fmt(f),
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

#[cfg(test)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum RunnableTestKind {
    Test,
    TestMod,
    DocTest,
    Bench,
    Bin,
}

impl Runnable {
    // test package::module::testname
    pub fn label(&self, target: Option<String>) -> String {
        match &self.kind {
            RunnableKind::Test { test_id, .. } => format!("test {test_id}"),
            RunnableKind::TestMod { path } => format!("test-mod {path}"),
            RunnableKind::Bench { test_id } => format!("bench {test_id}"),
            RunnableKind::DocTest { test_id, .. } => format!("doctest {test_id}"),
            RunnableKind::Bin => {
                target.map_or_else(|| "run binary".to_owned(), |t| format!("run {t}"))
            }
        }
    }

    pub fn title(&self) -> String {
        let mut s = String::from("▶\u{fe0e} Run ");
        if self.use_name_in_title {
            format_to!(s, "{}", self.nav.name);
            if !matches!(self.kind, RunnableKind::Bin) {
                s.push(' ');
            }
        }
        let suffix = match &self.kind {
            RunnableKind::TestMod { .. } => "Tests",
            RunnableKind::Test { .. } => "Test",
            RunnableKind::DocTest { .. } => "Doctest",
            RunnableKind::Bench { .. } => "Bench",
            RunnableKind::Bin => return s,
        };
        s.push_str(suffix);
        s
    }

    #[cfg(test)]
    fn test_kind(&self) -> RunnableTestKind {
        match &self.kind {
            RunnableKind::TestMod { .. } => RunnableTestKind::TestMod,
            RunnableKind::Test { .. } => RunnableTestKind::Test,
            RunnableKind::DocTest { .. } => RunnableTestKind::DocTest,
            RunnableKind::Bench { .. } => RunnableTestKind::Bench,
            RunnableKind::Bin => RunnableTestKind::Bin,
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
// | VS Code | **rust-analyzer: Run**
// |===
// image::https://user-images.githubusercontent.com/48062697/113065583-055aae80-91b1-11eb-958f-d67efcaf6a2f.gif[]
pub(crate) fn runnables(db: &RootDatabase, file_id: FileId) -> Vec<Runnable> {
    let sema = Semantics::new(db);

    let mut res = Vec::new();
    // Record all runnables that come from macro expansions here instead.
    // In case an expansion creates multiple runnables we want to name them to avoid emitting a bunch of equally named runnables.
    let mut in_macro_expansion = FxHashMap::<hir::HirFileId, Vec<Runnable>>::default();
    let mut add_opt = |runnable: Option<Runnable>, def| {
        if let Some(runnable) = runnable.filter(|runnable| {
            always!(
                runnable.nav.file_id == file_id,
                "tried adding a runnable pointing to a different file: {:?} for {:?}",
                runnable.kind,
                file_id
            )
        }) {
            if let Some(def) = def {
                let file_id = match def {
                    Definition::Module(it) => it.declaration_source(db).map(|src| src.file_id),
                    Definition::Function(it) => it.source(db).map(|src| src.file_id),
                    _ => None,
                };
                if let Some(file_id) = file_id.filter(|file| file.macro_file().is_some()) {
                    in_macro_expansion.entry(file_id).or_default().push(runnable);
                    return;
                }
            }
            res.push(runnable);
        }
    };
    visit_file_defs(&sema, file_id, &mut |def| {
        let runnable = match def {
            Definition::Module(it) => runnable_mod(&sema, it),
            Definition::Function(it) => runnable_fn(&sema, it),
            Definition::SelfType(impl_) => runnable_impl(&sema, &impl_),
            _ => None,
        };
        add_opt(
            runnable
                .or_else(|| module_def_doctest(sema.db, def))
                // #[macro_export] mbe macros are declared in the root, while their definition may reside in a different module
                .filter(|it| it.nav.file_id == file_id),
            Some(def),
        );
        if let Definition::SelfType(impl_) = def {
            impl_.items(db).into_iter().for_each(|assoc| {
                let runnable = match assoc {
                    hir::AssocItem::Function(it) => {
                        runnable_fn(&sema, it).or_else(|| module_def_doctest(sema.db, it.into()))
                    }
                    hir::AssocItem::Const(it) => module_def_doctest(sema.db, it.into()),
                    hir::AssocItem::TypeAlias(it) => module_def_doctest(sema.db, it.into()),
                };
                add_opt(runnable, Some(assoc.into()))
            });
        }
    });

    sema.to_module_defs(file_id)
        .map(|it| runnable_mod_outline_definition(&sema, it))
        .for_each(|it| add_opt(it, None));

    res.extend(in_macro_expansion.into_iter().flat_map(|(_, runnables)| {
        let use_name_in_title = runnables.len() != 1;
        runnables.into_iter().map(move |mut r| {
            r.use_name_in_title = use_name_in_title;
            r
        })
    }));
    res
}

// Feature: Related Tests
//
// Provides a sneak peek of all tests where the current item is used.
//
// The simplest way to use this feature is via the context menu. Right-click on
// the selected item. The context menu opens. Select **Peek Related Tests**.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Peek Related Tests**
// |===
pub(crate) fn related_tests(
    db: &RootDatabase,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Vec<Runnable> {
    let sema = Semantics::new(db);
    let mut res: FxHashSet<Runnable> = FxHashSet::default();
    let syntax = sema.parse(position.file_id).syntax().clone();

    find_related_tests(&sema, &syntax, position, search_scope, &mut res);

    res.into_iter().collect()
}

fn find_related_tests(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    search_scope: Option<SearchScope>,
    tests: &mut FxHashSet<Runnable>,
) {
    // FIXME: why is this using references::find_defs, this should use ide_db::search
    let defs = match references::find_defs(sema, syntax, position.offset) {
        Some(defs) => defs,
        None => return,
    };
    for def in defs {
        let defs = def
            .usages(sema)
            .set_scope(search_scope.as_ref())
            .all()
            .references
            .into_values()
            .flatten();
        for ref_ in defs {
            let name_ref = match ref_.name {
                FileReferenceNode::NameRef(name_ref) => name_ref,
                _ => continue,
            };
            if let Some(fn_def) =
                sema.ancestors_with_macros(name_ref.syntax().clone()).find_map(ast::Fn::cast)
            {
                if let Some(runnable) = as_test_runnable(sema, &fn_def) {
                    // direct test
                    tests.insert(runnable);
                } else if let Some(module) = parent_test_module(sema, &fn_def) {
                    // indirect test
                    find_related_tests_in_module(sema, syntax, &fn_def, &module, tests);
                }
            }
        }
    }
}

fn find_related_tests_in_module(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    fn_def: &ast::Fn,
    parent_module: &hir::Module,
    tests: &mut FxHashSet<Runnable>,
) {
    let fn_name = match fn_def.name() {
        Some(it) => it,
        _ => return,
    };
    let mod_source = parent_module.definition_source(sema.db);
    let range = match &mod_source.value {
        hir::ModuleSource::Module(m) => m.syntax().text_range(),
        hir::ModuleSource::BlockExpr(b) => b.syntax().text_range(),
        hir::ModuleSource::SourceFile(f) => f.syntax().text_range(),
    };

    let file_id = mod_source.file_id.original_file(sema.db);
    let mod_scope = SearchScope::file_range(FileRange { file_id, range });
    let fn_pos = FilePosition { file_id, offset: fn_name.syntax().text_range().start() };
    find_related_tests(sema, syntax, fn_pos, Some(mod_scope), tests)
}

fn as_test_runnable(sema: &Semantics<'_, RootDatabase>, fn_def: &ast::Fn) -> Option<Runnable> {
    if test_related_attribute(fn_def).is_some() {
        let function = sema.to_def(fn_def)?;
        runnable_fn(sema, function)
    } else {
        None
    }
}

fn parent_test_module(sema: &Semantics<'_, RootDatabase>, fn_def: &ast::Fn) -> Option<hir::Module> {
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

pub(crate) fn runnable_fn(
    sema: &Semantics<'_, RootDatabase>,
    def: hir::Function,
) -> Option<Runnable> {
    let kind = if def.is_main(sema.db) {
        RunnableKind::Bin
    } else {
        let test_id = || {
            let canonical_path = {
                let def: hir::ModuleDef = def.into();
                def.canonical_path(sema.db)
            };
            canonical_path
                .map(TestId::Path)
                .unwrap_or(TestId::Name(def.name(sema.db).to_smol_str()))
        };

        if def.is_test(sema.db) {
            let attr = TestAttr::from_fn(sema.db, def);
            RunnableKind::Test { test_id: test_id(), attr }
        } else if def.is_bench(sema.db) {
            RunnableKind::Bench { test_id: test_id() }
        } else {
            return None;
        }
    };

    let nav = NavigationTarget::from_named(
        sema.db,
        def.source(sema.db)?.as_ref().map(|it| it as &dyn ast::HasName),
        SymbolKind::Function,
    )
    .call_site();
    let cfg = def.attrs(sema.db).cfg();
    Some(Runnable { use_name_in_title: false, nav, kind, cfg })
}

pub(crate) fn runnable_mod(
    sema: &Semantics<'_, RootDatabase>,
    def: hir::Module,
) -> Option<Runnable> {
    if !has_test_function_or_multiple_test_submodules(sema, &def) {
        return None;
    }
    let path = def
        .path_to_root(sema.db)
        .into_iter()
        .rev()
        .filter_map(|it| it.name(sema.db))
        .map(|it| it.display(sema.db).to_string())
        .join("::");

    let attrs = def.attrs(sema.db);
    let cfg = attrs.cfg();
    let nav = NavigationTarget::from_module_to_decl(sema.db, def).call_site();
    Some(Runnable { use_name_in_title: false, nav, kind: RunnableKind::TestMod { path }, cfg })
}

pub(crate) fn runnable_impl(
    sema: &Semantics<'_, RootDatabase>,
    def: &hir::Impl,
) -> Option<Runnable> {
    let attrs = def.attrs(sema.db);
    if !has_runnable_doc_test(&attrs) {
        return None;
    }
    let cfg = attrs.cfg();
    let nav = def.try_to_nav(sema.db)?.call_site();
    let ty = def.self_ty(sema.db);
    let adt_name = ty.as_adt()?.name(sema.db);
    let mut ty_args = ty.generic_parameters(sema.db).peekable();
    let params = if ty_args.peek().is_some() {
        format!("<{}>", ty_args.format_with(",", |ty, cb| cb(&ty)))
    } else {
        String::new()
    };
    let mut test_id = format!("{}{params}", adt_name.display(sema.db));
    test_id.retain(|c| c != ' ');
    let test_id = TestId::Path(test_id);

    Some(Runnable { use_name_in_title: false, nav, kind: RunnableKind::DocTest { test_id }, cfg })
}

/// Creates a test mod runnable for outline modules at the top of their definition.
fn runnable_mod_outline_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: hir::Module,
) -> Option<Runnable> {
    if !has_test_function_or_multiple_test_submodules(sema, &def) {
        return None;
    }
    let path = def
        .path_to_root(sema.db)
        .into_iter()
        .rev()
        .filter_map(|it| it.name(sema.db))
        .map(|it| it.display(sema.db).to_string())
        .join("::");

    let attrs = def.attrs(sema.db);
    let cfg = attrs.cfg();
    match def.definition_source(sema.db).value {
        hir::ModuleSource::SourceFile(_) => Some(Runnable {
            use_name_in_title: false,
            nav: def.to_nav(sema.db).call_site(),
            kind: RunnableKind::TestMod { path },
            cfg,
        }),
        _ => None,
    }
}

fn module_def_doctest(db: &RootDatabase, def: Definition) -> Option<Runnable> {
    let attrs = match def {
        Definition::Module(it) => it.attrs(db),
        Definition::Function(it) => it.attrs(db),
        Definition::Adt(it) => it.attrs(db),
        Definition::Variant(it) => it.attrs(db),
        Definition::Const(it) => it.attrs(db),
        Definition::Static(it) => it.attrs(db),
        Definition::Trait(it) => it.attrs(db),
        Definition::TraitAlias(it) => it.attrs(db),
        Definition::TypeAlias(it) => it.attrs(db),
        Definition::Macro(it) => it.attrs(db),
        Definition::SelfType(it) => it.attrs(db),
        _ => return None,
    };
    if !has_runnable_doc_test(&attrs) {
        return None;
    }
    let def_name = def.name(db)?;
    let path = (|| {
        let mut path = String::new();
        def.canonical_module_path(db)?
            .flat_map(|it| it.name(db))
            .for_each(|name| format_to!(path, "{}::", name.display(db)));
        // This probably belongs to canonical_path?
        if let Some(assoc_item) = def.as_assoc_item(db) {
            if let Some(ty) = assoc_item.implementing_ty(db) {
                if let Some(adt) = ty.as_adt() {
                    let name = adt.name(db);
                    let mut ty_args = ty.generic_parameters(db).peekable();
                    format_to!(path, "{}", name.display(db));
                    if ty_args.peek().is_some() {
                        format_to!(path, "<{}>", ty_args.format_with(",", |ty, cb| cb(&ty)));
                    }
                    format_to!(path, "::{}", def_name.display(db));
                    path.retain(|c| c != ' ');
                    return Some(path);
                }
            }
        }
        format_to!(path, "{}", def_name.display(db));
        Some(path)
    })();

    let test_id = path.map_or_else(|| TestId::Name(def_name.to_smol_str()), TestId::Path);

    let mut nav = match def {
        Definition::Module(def) => NavigationTarget::from_module_to_decl(db, def),
        def => def.try_to_nav(db)?,
    }
    .call_site();
    nav.focus_range = None;
    nav.description = None;
    nav.docs = None;
    nav.kind = None;
    let res = Runnable {
        use_name_in_title: false,
        nav,
        kind: RunnableKind::DocTest { test_id },
        cfg: attrs.cfg(),
    };
    Some(res)
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TestAttr {
    pub ignore: bool,
}

impl TestAttr {
    fn from_fn(db: &dyn HirDatabase, fn_def: hir::Function) -> TestAttr {
        TestAttr { ignore: fn_def.is_ignore(db) }
    }
}

const RUSTDOC_FENCES: [&str; 2] = ["```", "~~~"];
const RUSTDOC_CODE_BLOCK_ATTRIBUTES_RUNNABLE: &[&str] =
    &["", "rust", "should_panic", "edition2015", "edition2018", "edition2021"];

fn has_runnable_doc_test(attrs: &hir::Attrs) -> bool {
    docs_from_attrs(attrs).map_or(false, |doc| {
        let mut in_code_block = false;

        for line in doc.lines() {
            if let Some(header) =
                RUSTDOC_FENCES.into_iter().find_map(|fence| line.strip_prefix(fence))
            {
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
    sema: &Semantics<'_, RootDatabase>,
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

    use crate::fixture;

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let mut runnables = analysis.runnables(position.file_id).unwrap();
        runnables.sort_by_key(|it| (it.nav.full_range.start(), it.nav.name.clone()));

        let result = runnables
            .into_iter()
            .map(|runnable| {
                let mut a = format!("({:?}, {:?}", runnable.test_kind(), runnable.nav);
                if runnable.use_name_in_title {
                    a.push_str(", true");
                }
                if let Some(cfg) = runnable.cfg {
                    a.push_str(&format!(", {cfg:?}"));
                }
                a.push(')');
                a
            })
            .collect::<Vec<_>>();
        expect.assert_debug_eq(&result);
    }

    fn check_tests(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let tests = analysis.related_tests(position, None).unwrap();
        let navigation_targets = tests.into_iter().map(|runnable| runnable.nav).collect::<Vec<_>>();
        expect.assert_debug_eq(&navigation_targets);
    }

    #[test]
    fn test_runnables() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

#[export_name = "main"]
fn __cortex_m_rt_main_trampoline() {}

#[test]
fn test_foo() {}

#[::core::prelude::v1::test]
fn test_full_path() {}

#[test]
#[ignore]
fn test_foo() {}

#[bench]
fn bench() {}

mod not_a_root {
    fn main() {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..253, name: \"\", kind: Module })",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 15..76, focus_range: 42..71, name: \"__cortex_m_rt_main_trampoline\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 78..102, focus_range: 89..97, name: \"test_foo\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 104..155, focus_range: 136..150, name: \"test_full_path\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 157..191, focus_range: 178..186, name: \"test_foo\", kind: Function })",
                    "(Bench, NavigationTarget { file_id: FileId(0), full_range: 193..215, focus_range: 205..210, name: \"bench\", kind: Function })",
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

/**
```rust
let z = 55;
```
*/
fn should_have_no_runnable_3() {}

/**
    ```rust
    let z = 55;
    ```
*/
fn should_have_no_runnable_4() {}

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

/// ```
/// let x = 5;
/// ```
impl StructWithRunnable {}

trait Test {
    fn test() -> usize {
        5usize
    }
}

/// ```
/// let x = 5;
/// ```
impl Test for StructWithRunnable {}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 15..74, name: \"should_have_runnable\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 76..148, name: \"should_have_runnable_1\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 150..254, name: \"should_have_runnable_2\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 256..320, name: \"should_have_no_runnable_3\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 322..398, name: \"should_have_no_runnable_4\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 900..965, name: \"StructWithRunnable\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 967..1024, focus_range: 1003..1021, name: \"impl\", kind: Impl })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 1088..1154, focus_range: 1133..1151, name: \"impl\", kind: Impl })",
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
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 44..98, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl_with_lifetime() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

struct Data<'a>;
impl Data<'a> {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 52..106, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl_with_lifetime_and_types() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

struct Data<'a, T, U>;
impl<T, U> Data<'a, T, U> {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 70..124, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl_with_const() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

struct Data<const N: usize>;
impl<const N: usize> Data<N> {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 79..133, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl_with_lifetime_types_and_const() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

struct Data<'a, T, const N: usize>;
impl<'a, T, const N: usize> Data<'a, T, N> {
    /// ```
    /// let x = 5;
    /// ```
    fn foo() {}
}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 100..154, name: \"foo\" })",
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
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 1..51, focus_range: 5..13, name: \"test_mod\", kind: Module, description: \"mod test_mod\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 20..49, focus_range: 35..44, name: \"test_foo1\", kind: Function })",
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
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 22..323, focus_range: 26..40, name: \"nested_tests_0\", kind: Module, description: \"mod nested_tests_0\" })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 51..192, focus_range: 55..69, name: \"nested_tests_1\", kind: Module, description: \"mod nested_tests_1\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 84..126, focus_range: 107..121, name: \"nested_test_11\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 140..182, focus_range: 163..177, name: \"nested_test_12\", kind: Function })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 202..286, focus_range: 206..220, name: \"nested_tests_2\", kind: Module, description: \"mod nested_tests_2\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 235..276, focus_range: 258..271, name: \"nested_test_2\", kind: Function })",
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
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..51, name: \"\", kind: Module })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 1..50, focus_range: 36..45, name: \"test_foo1\", kind: Function }, Atom(KeyValue { key: \"feature\", value: \"foo\" }))",
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
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..73, name: \"\", kind: Module })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 1..72, focus_range: 58..67, name: \"test_foo1\", kind: Function }, All([Atom(KeyValue { key: \"feature\", value: \"foo\" }), Atom(KeyValue { key: \"feature\", value: \"bar\" })]))",
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
            expect![[r#"
                [
                    "(DocTest, NavigationTarget { file_id: FileId(1), full_range: 27..81, name: \"foo\" })",
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
        fn foo_test() {}
    }
}
macro_rules! gen2 {
    () => {
        mod tests2 {
            #[test]
            fn foo_test2() {}
        }
    }
}
macro_rules! gen_main {
    () => {
        fn main() {}
    }
}
mod tests {
    gen!();
}
gen2!();
gen_main!();
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..315, name: \"\", kind: Module })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 267..292, focus_range: 271..276, name: \"tests\", kind: Module, description: \"mod tests\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 283..290, name: \"foo_test\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 293..301, name: \"foo_test2\", kind: Function }, true)",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 293..301, name: \"tests2\", kind: Module, description: \"mod tests2\" }, true)",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 302..314, name: \"main\", kind: Function })",
                ]
            "#]],
        );
    }

    #[test]
    fn big_mac() {
        check(
            r#"
//- /lib.rs
$0
macro_rules! foo {
    () => {
        mod foo_tests {
            #[test]
            fn foo0() {}
            #[test]
            fn foo1() {}
            #[test]
            fn foo2() {}
        }
    };
}
foo!();
"#,
            expect![[r#"
                [
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo0\", kind: Function }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo1\", kind: Function }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo2\", kind: Function }, true)",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo_tests\", kind: Module, description: \"mod foo_tests\" }, true)",
                ]
            "#]],
        );
    }

    #[test]
    fn dont_recurse_in_outline_submodules() {
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
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
    fn outline_submodule1() {
        check(
            r#"
//- /lib.rs
$0
mod m;
//- /m.rs
#[test]
fn t0() {}
#[test]
fn t1() {}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 1..7, focus_range: 5..6, name: \"m\", kind: Module, description: \"mod m\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn outline_submodule2() {
        check(
            r#"
//- /lib.rs
mod m;
//- /m.rs
$0
#[test]
fn t0() {}
#[test]
fn t1() {}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(1), full_range: 0..39, name: \"m\", kind: Module })",
                    "(Test, NavigationTarget { file_id: FileId(1), full_range: 1..19, focus_range: 12..14, name: \"t0\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(1), full_range: 20..38, focus_range: 31..33, name: \"t1\", kind: Function })",
                ]
            "#]],
        );
    }

    #[test]
    fn attributed_module() {
        check(
            r#"
//- proc_macros: identity
//- /lib.rs
$0
#[proc_macros::identity]
mod module {
    #[test]
    fn t0() {}
    #[test]
    fn t1() {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 26..94, focus_range: 30..36, name: \"module\", kind: Module, description: \"mod module\" }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 43..65, focus_range: 58..60, name: \"t0\", kind: Function }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 70..92, focus_range: 85..87, name: \"t1\", kind: Function }, true)",
                ]
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
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 31..85,
                        focus_range: 46..54,
                        name: "foo_test",
                        kind: Function,
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
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 71..122,
                        focus_range: 86..94,
                        name: "foo_test",
                        kind: Function,
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
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 133..183,
                        focus_range: 148..156,
                        name: "foo_test",
                        kind: Function,
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
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 121..185,
                        focus_range: 136..145,
                        name: "foo2_test",
                        kind: Function,
                    },
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 52..115,
                        focus_range: 67..75,
                        name: "foo_test",
                        kind: Function,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_runnables_doc_test_in_impl_with_lifetime_type_const_value() {
        check(
            r#"
//- /lib.rs
$0
fn main() {}

struct Data<'a, A, const B: usize, C, const D: u32>;
impl<A, C, const D: u32> Data<'a, A, 12, C, D> {
    /// ```
    /// ```
    fn foo() {}
}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 121..156, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn doc_test_type_params() {
        check(
            r#"
//- /lib.rs
$0
struct Foo<T, U>;

/// ```
/// ```
impl<T, U> Foo<T, U> {
    /// ```rust
    /// ````
    fn t() {}
}

/// ```
/// ```
impl Foo<Foo<(), ()>, ()> {
    /// ```
    /// ```
    fn t() {}
}
"#,
            expect![[r#"
                [
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 20..103, focus_range: 47..56, name: \"impl\", kind: Impl })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 63..101, name: \"t\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 105..188, focus_range: 126..146, name: \"impl\", kind: Impl })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 153..186, name: \"t\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn doc_test_macro_export_mbe() {
        check(
            r#"
//- /lib.rs
$0
mod foo;

//- /foo.rs
/// ```
/// fn foo() {
/// }
/// ```
#[macro_export]
macro_rules! foo {
    () => {

    };
}
"#,
            expect![[r#"
                []
            "#]],
        );
        check(
            r#"
//- /lib.rs
$0
/// ```
/// fn foo() {
/// }
/// ```
#[macro_export]
macro_rules! foo {
    () => {

    };
}
"#,
            expect![[r#"
                [
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 1..94, name: \"foo\" })",
                ]
            "#]],
        );
    }

    #[test]
    fn test_paths_with_raw_ident() {
        check(
            r#"
//- /lib.rs
$0
mod r#mod {
    #[test]
    fn r#fn() {}

    /// ```
    /// ```
    fn r#for() {}

    /// ```
    /// ```
    struct r#struct<r#type>(r#type);

    /// ```
    /// ```
    impl<r#type> r#struct<r#type> {
        /// ```
        /// ```
        fn r#fn() {}
    }

    enum r#enum {}
    impl r#struct<r#enum> {
        /// ```
        /// ```
        fn r#fn() {}
    }

    trait r#trait {}

    /// ```
    /// ```
    impl<T> r#trait for r#struct<T> {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 1..461, focus_range: 5..10, name: \"r#mod\", kind: Module, description: \"mod r#mod\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 17..41, focus_range: 32..36, name: \"r#fn\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 47..84, name: \"r#for\", container_name: \"r#mod\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 90..146, name: \"r#struct\", container_name: \"r#mod\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 152..266, focus_range: 189..205, name: \"impl\", kind: Impl })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 216..260, name: \"r#fn\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 323..367, name: \"r#fn\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 401..459, focus_range: 445..456, name: \"impl\", kind: Impl })",
                ]
            "#]],
        )
    }
}
