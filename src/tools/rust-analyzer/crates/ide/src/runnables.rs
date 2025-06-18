use std::{fmt, sync::OnceLock};

use arrayvec::ArrayVec;
use ast::HasName;
use cfg::{CfgAtom, CfgExpr};
use hir::{
    AsAssocItem, AttrsWithOwner, HasAttrs, HasCrate, HasSource, ModPath, Name, PathKind, Semantics,
    Symbol, db::HirDatabase, sym,
};
use ide_assists::utils::{has_test_related_attribute, test_related_attribute_syn};
use ide_db::{
    FilePosition, FxHashMap, FxIndexMap, FxIndexSet, RootDatabase, SymbolKind,
    base_db::RootQueryDb,
    defs::Definition,
    documentation::docs_from_attrs,
    helpers::visit_file_defs,
    search::{FileReferenceNode, SearchScope},
};
use itertools::Itertools;
use smallvec::SmallVec;
use span::{Edition, TextSize};
use stdx::format_to;
use syntax::{
    SmolStr, SyntaxNode, ToSmolStr,
    ast::{self, AstNode},
    format_smolstr,
};

use crate::{FileId, NavigationTarget, ToNav, TryToNav, references};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Runnable {
    pub use_name_in_title: bool,
    pub nav: NavigationTarget,
    pub kind: RunnableKind,
    pub cfg: Option<CfgExpr>,
    pub update_test: UpdateTest,
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
    TestMod { path: String },
    Test { test_id: TestId, attr: TestAttr },
    Bench { test_id: TestId },
    DocTest { test_id: TestId },
    Bin,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum RunnableDiscKind {
    TestMod,
    Test,
    DocTest,
    Bench,
    Bin,
}

impl RunnableKind {
    fn disc(&self) -> RunnableDiscKind {
        match self {
            RunnableKind::TestMod { .. } => RunnableDiscKind::TestMod,
            RunnableKind::Test { .. } => RunnableDiscKind::Test,
            RunnableKind::DocTest { .. } => RunnableDiscKind::DocTest,
            RunnableKind::Bench { .. } => RunnableDiscKind::Bench,
            RunnableKind::Bin => RunnableDiscKind::Bin,
        }
    }
}

impl Runnable {
    pub fn label(&self, target: Option<&str>) -> String {
        match &self.kind {
            RunnableKind::Test { test_id, .. } => format!("test {test_id}"),
            RunnableKind::TestMod { path } => format!("test-mod {path}"),
            RunnableKind::Bench { test_id } => format!("bench {test_id}"),
            RunnableKind::DocTest { test_id, .. } => format!("doctest {test_id}"),
            RunnableKind::Bin => {
                format!("run {}", target.unwrap_or("binary"))
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
}

// Feature: Run
//
// Shows a popup suggesting to run a test/benchmark/binary **at the current cursor
// location**. Super useful for repeatedly running just a single test. Do bind this
// to a shortcut!
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Run** |
//
// ![Run](https://user-images.githubusercontent.com/48062697/113065583-055aae80-91b1-11eb-958f-d67efcaf6a2f.gif)
pub(crate) fn runnables(db: &RootDatabase, file_id: FileId) -> Vec<Runnable> {
    let sema = Semantics::new(db);

    let mut res = Vec::new();
    // Record all runnables that come from macro expansions here instead.
    // In case an expansion creates multiple runnables we want to name them to avoid emitting a bunch of equally named runnables.
    let mut in_macro_expansion = FxIndexMap::<hir::HirFileId, Vec<Runnable>>::default();
    let mut add_opt = |runnable: Option<Runnable>, def| {
        if let Some(runnable) = runnable.filter(|runnable| runnable.nav.file_id == file_id) {
            if let Some(def) = def {
                let file_id = match def {
                    Definition::Module(it) => {
                        it.declaration_source_range(db).map(|src| src.file_id)
                    }
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
        add_opt(runnable.or_else(|| module_def_doctest(sema.db, def)), Some(def));
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

    sema.file_to_module_defs(file_id)
        .map(|it| runnable_mod_outline_definition(&sema, it))
        .for_each(|it| add_opt(it, None));

    res.extend(in_macro_expansion.into_iter().flat_map(|(_, runnables)| {
        let use_name_in_title = runnables.len() != 1;
        runnables.into_iter().map(move |mut r| {
            r.use_name_in_title = use_name_in_title;
            r
        })
    }));
    res.sort_by(cmp_runnables);
    res
}

// Feature: Related Tests
//
// Provides a sneak peek of all tests where the current item is used.
//
// The simplest way to use this feature is via the context menu. Right-click on
// the selected item. The context menu opens. Select **Peek Related Tests**.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Peek Related Tests** |
pub(crate) fn related_tests(
    db: &RootDatabase,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Vec<Runnable> {
    let sema = Semantics::new(db);
    let mut res: FxIndexSet<Runnable> = FxIndexSet::default();
    let syntax = sema.parse_guess_edition(position.file_id).syntax().clone();

    find_related_tests(&sema, &syntax, position, search_scope, &mut res);

    res.into_iter().sorted_by(cmp_runnables).collect()
}

fn cmp_runnables(
    Runnable { nav, kind, .. }: &Runnable,
    Runnable { nav: nav_b, kind: kind_b, .. }: &Runnable,
) -> std::cmp::Ordering {
    // full_range.start < focus_range.start < name, should give us a decent unique ordering
    nav.full_range
        .start()
        .cmp(&nav_b.full_range.start())
        .then_with(|| {
            let t_0 = || TextSize::from(0);
            nav.focus_range
                .map_or_else(t_0, |it| it.start())
                .cmp(&nav_b.focus_range.map_or_else(t_0, |it| it.start()))
        })
        .then_with(|| kind.disc().cmp(&kind_b.disc()))
        .then_with(|| nav.name.cmp(&nav_b.name))
}

fn find_related_tests(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    search_scope: Option<SearchScope>,
    tests: &mut FxIndexSet<Runnable>,
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
    tests: &mut FxIndexSet<Runnable>,
) {
    let fn_name = match fn_def.name() {
        Some(it) => it,
        _ => return,
    };
    let mod_source = parent_module.definition_source_range(sema.db);

    let file_id = mod_source.file_id.original_file(sema.db);
    let mod_scope = SearchScope::file_range(hir::FileRange { file_id, range: mod_source.value });
    let fn_pos = FilePosition {
        file_id: file_id.file_id(sema.db),
        offset: fn_name.syntax().text_range().start(),
    };
    find_related_tests(sema, syntax, fn_pos, Some(mod_scope), tests)
}

fn as_test_runnable(sema: &Semantics<'_, RootDatabase>, fn_def: &ast::Fn) -> Option<Runnable> {
    if test_related_attribute_syn(fn_def).is_some() {
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

        if has_test_function_or_multiple_test_submodules(sema, &module, false) {
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
    let edition = def.krate(sema.db).edition(sema.db);
    let under_cfg_test = has_cfg_test(def.module(sema.db).attrs(sema.db));
    let kind = if !under_cfg_test && def.is_main(sema.db) {
        RunnableKind::Bin
    } else {
        let test_id = || {
            let canonical_path = {
                let def: hir::ModuleDef = def.into();
                def.canonical_path(sema.db, edition)
            };
            canonical_path
                .map(TestId::Path)
                .unwrap_or(TestId::Name(def.name(sema.db).display_no_db(edition).to_smolstr()))
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

    let fn_source = sema.source(def)?;
    let nav = NavigationTarget::from_named(
        sema.db,
        fn_source.as_ref().map(|it| it as &dyn ast::HasName),
        SymbolKind::Function,
    )
    .call_site();

    let file_range = fn_source.syntax().original_file_range_with_macro_call_body(sema.db);
    let update_test =
        UpdateTest::find_snapshot_macro(sema, &fn_source.file_syntax(sema.db), file_range);

    let cfg = def.attrs(sema.db).cfg();
    Some(Runnable { use_name_in_title: false, nav, kind, cfg, update_test })
}

pub(crate) fn runnable_mod(
    sema: &Semantics<'_, RootDatabase>,
    def: hir::Module,
) -> Option<Runnable> {
    if !has_test_function_or_multiple_test_submodules(sema, &def, has_cfg_test(def.attrs(sema.db)))
    {
        return None;
    }
    let path = def
        .path_to_root(sema.db)
        .into_iter()
        .rev()
        .filter_map(|module| {
            module.name(sema.db).map(|mod_name| {
                mod_name.display(sema.db, module.krate().edition(sema.db)).to_string()
            })
        })
        .join("::");

    let attrs = def.attrs(sema.db);
    let cfg = attrs.cfg();
    let nav = NavigationTarget::from_module_to_decl(sema.db, def).call_site();

    let module_source = sema.module_definition_node(def);
    let module_syntax = module_source.file_syntax(sema.db);
    let file_range = hir::FileRange {
        file_id: module_source.file_id.original_file(sema.db),
        range: module_syntax.text_range(),
    };
    let update_test = UpdateTest::find_snapshot_macro(sema, &module_syntax, file_range);

    Some(Runnable {
        use_name_in_title: false,
        nav,
        kind: RunnableKind::TestMod { path },
        cfg,
        update_test,
    })
}

pub(crate) fn runnable_impl(
    sema: &Semantics<'_, RootDatabase>,
    def: &hir::Impl,
) -> Option<Runnable> {
    let display_target = def.module(sema.db).krate().to_display_target(sema.db);
    let edition = display_target.edition;
    let attrs = def.attrs(sema.db);
    if !has_runnable_doc_test(&attrs) {
        return None;
    }
    let cfg = attrs.cfg();
    let nav = def.try_to_nav(sema.db)?.call_site();
    let ty = def.self_ty(sema.db);
    let adt_name = ty.as_adt()?.name(sema.db);
    let mut ty_args = ty.generic_parameters(sema.db, display_target).peekable();
    let params = if ty_args.peek().is_some() {
        format!("<{}>", ty_args.format_with(",", |ty, cb| cb(&ty)))
    } else {
        String::new()
    };
    let mut test_id = format!("{}{params}", adt_name.display(sema.db, edition));
    test_id.retain(|c| c != ' ');
    let test_id = TestId::Path(test_id);

    let impl_source = sema.source(*def)?;
    let impl_syntax = impl_source.syntax();
    let file_range = impl_syntax.original_file_range_with_macro_call_body(sema.db);
    let update_test =
        UpdateTest::find_snapshot_macro(sema, &impl_syntax.file_syntax(sema.db), file_range);

    Some(Runnable {
        use_name_in_title: false,
        nav,
        kind: RunnableKind::DocTest { test_id },
        cfg,
        update_test,
    })
}

fn has_cfg_test(attrs: AttrsWithOwner) -> bool {
    attrs.cfgs().any(|cfg| matches!(&cfg, CfgExpr::Atom(CfgAtom::Flag(s)) if *s == sym::test))
}

/// Creates a test mod runnable for outline modules at the top of their definition.
fn runnable_mod_outline_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: hir::Module,
) -> Option<Runnable> {
    def.as_source_file_id(sema.db)?;

    if !has_test_function_or_multiple_test_submodules(sema, &def, has_cfg_test(def.attrs(sema.db)))
    {
        return None;
    }
    let path = def
        .path_to_root(sema.db)
        .into_iter()
        .rev()
        .filter_map(|module| {
            module.name(sema.db).map(|mod_name| {
                mod_name.display(sema.db, module.krate().edition(sema.db)).to_string()
            })
        })
        .join("::");

    let attrs = def.attrs(sema.db);
    let cfg = attrs.cfg();

    let mod_source = sema.module_definition_node(def);
    let mod_syntax = mod_source.file_syntax(sema.db);
    let file_range = hir::FileRange {
        file_id: mod_source.file_id.original_file(sema.db),
        range: mod_syntax.text_range(),
    };
    let update_test = UpdateTest::find_snapshot_macro(sema, &mod_syntax, file_range);

    Some(Runnable {
        use_name_in_title: false,
        nav: def.to_nav(sema.db).call_site(),
        kind: RunnableKind::TestMod { path },
        cfg,
        update_test,
    })
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
    let krate = def.krate(db);
    let edition = krate.map(|it| it.edition(db)).unwrap_or(Edition::CURRENT);
    let display_target = krate
        .unwrap_or_else(|| (*db.all_crates().last().expect("no crate graph present")).into())
        .to_display_target(db);
    if !has_runnable_doc_test(&attrs) {
        return None;
    }
    let def_name = def.name(db)?;
    let path = (|| {
        let mut path = String::new();
        def.canonical_module_path(db)?
            .flat_map(|it| it.name(db))
            .for_each(|name| format_to!(path, "{}::", name.display(db, edition)));
        // This probably belongs to canonical_path?
        if let Some(assoc_item) = def.as_assoc_item(db) {
            if let Some(ty) = assoc_item.implementing_ty(db) {
                if let Some(adt) = ty.as_adt() {
                    let name = adt.name(db);
                    let mut ty_args = ty.generic_parameters(db, display_target).peekable();
                    format_to!(path, "{}", name.display(db, edition));
                    if ty_args.peek().is_some() {
                        format_to!(path, "<{}>", ty_args.format_with(",", |ty, cb| cb(&ty)));
                    }
                    format_to!(path, "::{}", def_name.display(db, edition));
                    path.retain(|c| c != ' ');
                    return Some(path);
                }
            }
        }
        format_to!(path, "{}", def_name.display(db, edition));
        Some(path)
    })();

    let test_id = path
        .map_or_else(|| TestId::Name(def_name.display_no_db(edition).to_smolstr()), TestId::Path);

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
        update_test: UpdateTest::default(),
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

fn has_runnable_doc_test(attrs: &hir::Attrs) -> bool {
    const RUSTDOC_FENCES: [&str; 2] = ["```", "~~~"];
    const RUSTDOC_CODE_BLOCK_ATTRIBUTES_RUNNABLE: &[&str] =
        &["", "rust", "should_panic", "edition2015", "edition2018", "edition2021"];

    docs_from_attrs(attrs).is_some_and(|doc| {
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
    consider_exported_main: bool,
) -> bool {
    let mut number_of_test_submodules = 0;

    for item in module.declarations(sema.db) {
        match item {
            hir::ModuleDef::Function(f) => {
                if has_test_related_attribute(&f.attrs(sema.db)) {
                    return true;
                }
                if consider_exported_main && f.exported_main(sema.db) {
                    // an exported main in a test module can be considered a test wrt to custom test
                    // runners
                    return true;
                }
            }
            hir::ModuleDef::Module(submodule) => {
                if has_test_function_or_multiple_test_submodules(
                    sema,
                    &submodule,
                    consider_exported_main,
                ) {
                    number_of_test_submodules += 1;
                }
            }
            _ => (),
        }
    }

    number_of_test_submodules > 1
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UpdateTest {
    pub expect_test: bool,
    pub insta: bool,
    pub snapbox: bool,
}

static SNAPSHOT_TEST_MACROS: OnceLock<FxHashMap<&str, Vec<ModPath>>> = OnceLock::new();

impl UpdateTest {
    const EXPECT_CRATE: &str = "expect_test";
    const EXPECT_MACROS: &[&str] = &["expect", "expect_file"];

    const INSTA_CRATE: &str = "insta";
    const INSTA_MACROS: &[&str] = &[
        "assert_snapshot",
        "assert_debug_snapshot",
        "assert_display_snapshot",
        "assert_json_snapshot",
        "assert_yaml_snapshot",
        "assert_ron_snapshot",
        "assert_toml_snapshot",
        "assert_csv_snapshot",
        "assert_compact_json_snapshot",
        "assert_compact_debug_snapshot",
        "assert_binary_snapshot",
    ];

    const SNAPBOX_CRATE: &str = "snapbox";
    const SNAPBOX_MACROS: &[&str] = &["assert_data_eq", "file", "str"];

    fn find_snapshot_macro(
        sema: &Semantics<'_, RootDatabase>,
        scope: &SyntaxNode,
        file_range: hir::FileRange,
    ) -> Self {
        fn init<'a>(
            krate_name: &'a str,
            paths: &[&str],
            map: &mut FxHashMap<&'a str, Vec<ModPath>>,
        ) {
            let mut res = Vec::with_capacity(paths.len());
            let krate = Name::new_symbol_root(Symbol::intern(krate_name));
            for path in paths {
                let segments = [krate.clone(), Name::new_symbol_root(Symbol::intern(path))];
                let mod_path = ModPath::from_segments(PathKind::Abs, segments);
                res.push(mod_path);
            }
            map.insert(krate_name, res);
        }

        let mod_paths = SNAPSHOT_TEST_MACROS.get_or_init(|| {
            let mut map = FxHashMap::default();
            init(Self::EXPECT_CRATE, Self::EXPECT_MACROS, &mut map);
            init(Self::INSTA_CRATE, Self::INSTA_MACROS, &mut map);
            init(Self::SNAPBOX_CRATE, Self::SNAPBOX_MACROS, &mut map);
            map
        });

        let search_scope = SearchScope::file_range(file_range);
        let find_macro = |paths: &[ModPath]| {
            for path in paths {
                let Some(items) = sema.resolve_mod_path(scope, path) else {
                    continue;
                };
                for item in items {
                    if let hir::ItemInNs::Macros(makro) = item {
                        if Definition::Macro(makro)
                            .usages(sema)
                            .in_scope(&search_scope)
                            .at_least_one()
                        {
                            return true;
                        }
                    }
                }
            }
            false
        };

        UpdateTest {
            expect_test: find_macro(mod_paths.get(Self::EXPECT_CRATE).unwrap()),
            insta: find_macro(mod_paths.get(Self::INSTA_CRATE).unwrap()),
            snapbox: find_macro(mod_paths.get(Self::SNAPBOX_CRATE).unwrap()),
        }
    }

    pub fn label(&self) -> Option<SmolStr> {
        let mut builder: SmallVec<[_; 3]> = SmallVec::new();
        if self.expect_test {
            builder.push("Expect");
        }
        if self.insta {
            builder.push("Insta");
        }
        if self.snapbox {
            builder.push("Snapbox");
        }

        let res: SmolStr = builder.join(" + ").into();
        if res.is_empty() {
            None
        } else {
            Some(format_smolstr!("↺\u{fe0e} Update Tests ({res})"))
        }
    }

    pub fn env(&self) -> ArrayVec<(&str, &str), 3> {
        let mut env = ArrayVec::new();
        if self.expect_test {
            env.push(("UPDATE_EXPECT", "1"));
        }
        if self.insta {
            env.push(("INSTA_UPDATE", "always"));
        }
        if self.snapbox {
            env.push(("SNAPSHOTS", "overwrite"));
        }
        env
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let result = analysis
            .runnables(position.file_id)
            .unwrap()
            .into_iter()
            .map(|runnable| {
                let mut a = format!("({:?}, {:?}", runnable.kind.disc(), runnable.nav);
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

    fn check_tests(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
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

#[unsafe(export_name = "main")]
fn __cortex_m_rt_main_trampoline_unsafe() {}

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
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..331, name: \"\", kind: Module })",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 1..13, focus_range: 4..8, name: \"main\", kind: Function })",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 15..76, focus_range: 42..71, name: \"__cortex_m_rt_main_trampoline\", kind: Function })",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 78..154, focus_range: 113..149, name: \"__cortex_m_rt_main_trampoline_unsafe\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 156..180, focus_range: 167..175, name: \"test_foo\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 182..233, focus_range: 214..228, name: \"test_full_path\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 235..269, focus_range: 256..264, name: \"test_foo\", kind: Function })",
                    "(Bench, NavigationTarget { file_id: FileId(0), full_range: 271..293, focus_range: 283..288, name: \"bench\", kind: Function })",
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
macro_rules! generate {
    () => {
        #[test]
        fn foo_test() {}
    }
}
macro_rules! generate2 {
    () => {
        mod tests2 {
            #[test]
            fn foo_test2() {}
        }
    }
}
macro_rules! generate_main {
    () => {
        fn main() {}
    }
}
mod tests {
    generate!();
}
generate2!();
generate_main!();
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 0..345, name: \"\", kind: Module })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 282..312, focus_range: 286..291, name: \"tests\", kind: Module, description: \"mod tests\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 298..310, name: \"foo_test\", kind: Function })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 313..326, name: \"tests2\", kind: Module, description: \"mod tests2\" }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 313..326, name: \"foo_test2\", kind: Function }, true)",
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 327..344, name: \"main\", kind: Function })",
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
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo_tests\", kind: Module, description: \"mod foo_tests\" }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo0\", kind: Function }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo1\", kind: Function }, true)",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 210..217, name: \"foo2\", kind: Function }, true)",
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
                        full_range: 52..115,
                        focus_range: 67..75,
                        name: "foo_test",
                        kind: Function,
                    },
                    NavigationTarget {
                        file_id: FileId(
                            0,
                        ),
                        full_range: 121..185,
                        focus_range: 136..145,
                        name: "foo2_test",
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

    #[test]
    fn exported_main_is_test_in_cfg_test_mod() {
        check(
            r#"
//- /lib.rs crate:foo cfg:test
$0
mod not_a_test_module_inline {
    #[export_name = "main"]
    fn exp_main() {}
}
#[cfg(test)]
mod test_mod_inline {
    #[export_name = "main"]
    fn exp_main() {}
}
mod not_a_test_module;
#[cfg(test)]
mod test_mod;
//- /not_a_test_module.rs
#[export_name = "main"]
fn exp_main() {}
//- /test_mod.rs
#[export_name = "main"]
fn exp_main() {}
"#,
            expect![[r#"
                [
                    "(Bin, NavigationTarget { file_id: FileId(0), full_range: 36..80, focus_range: 67..75, name: \"exp_main\", kind: Function })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 83..168, focus_range: 100..115, name: \"test_mod_inline\", kind: Module, description: \"mod test_mod_inline\" }, Atom(Flag(\"test\")))",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 192..218, focus_range: 209..217, name: \"test_mod\", kind: Module, description: \"mod test_mod\" }, Atom(Flag(\"test\")))",
                ]
            "#]],
        )
    }
}
