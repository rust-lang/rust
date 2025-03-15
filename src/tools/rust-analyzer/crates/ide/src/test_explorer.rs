//! Discovers tests

use hir::{Crate, Module, ModuleDef, Semantics};
use ide_db::base_db;
use ide_db::{FileId, RootDatabase, base_db::RootQueryDb};
use syntax::TextRange;

use crate::{NavigationTarget, Runnable, TryToNav, runnables::runnable_fn};

#[derive(Debug)]
pub enum TestItemKind {
    Crate(base_db::Crate),
    Module,
    Function,
}

#[derive(Debug)]
pub struct TestItem {
    pub id: String,
    pub kind: TestItemKind,
    pub label: String,
    pub parent: Option<String>,
    pub file: Option<FileId>,
    pub text_range: Option<TextRange>,
    pub runnable: Option<Runnable>,
}

pub(crate) fn discover_test_roots(db: &RootDatabase) -> Vec<TestItem> {
    db.all_crates()
        .iter()
        .copied()
        .filter(|&id| id.data(db).origin.is_local())
        .filter_map(|id| {
            let test_id = id.extra_data(db).display_name.as_ref()?.to_string();
            Some(TestItem {
                kind: TestItemKind::Crate(id),
                label: test_id.clone(),
                id: test_id,
                parent: None,
                file: None,
                text_range: None,
                runnable: None,
            })
        })
        .collect()
}

fn find_crate_by_id(db: &RootDatabase, crate_id: &str) -> Option<base_db::Crate> {
    // here, we use display_name as the crate id. This is not super ideal, but it works since we
    // only show tests for the local crates.
    db.all_crates().iter().copied().find(|&id| {
        id.data(db).origin.is_local()
            && id.extra_data(db).display_name.as_ref().is_some_and(|x| x.to_string() == crate_id)
    })
}

fn discover_tests_in_module(
    db: &RootDatabase,
    module: Module,
    prefix_id: String,
    only_in_this_file: bool,
) -> Vec<TestItem> {
    let sema = Semantics::new(db);

    let mut r = vec![];
    for c in module.children(db) {
        let module_name = c
            .name(db)
            .as_ref()
            .map(|n| n.as_str().to_owned())
            .unwrap_or_else(|| "[mod without name]".to_owned());
        let module_id = format!("{prefix_id}::{module_name}");
        let module_children = discover_tests_in_module(db, c, module_id.clone(), only_in_this_file);
        if !module_children.is_empty() {
            let nav = NavigationTarget::from_module_to_decl(sema.db, c).call_site;
            r.push(TestItem {
                id: module_id,
                kind: TestItemKind::Module,
                label: module_name,
                parent: Some(prefix_id.clone()),
                file: Some(nav.file_id),
                text_range: Some(nav.focus_or_full_range()),
                runnable: None,
            });
            if !only_in_this_file || c.is_inline(db) {
                r.extend(module_children);
            }
        }
    }
    for def in module.declarations(db) {
        let ModuleDef::Function(f) = def else {
            continue;
        };
        if !f.is_test(db) {
            continue;
        }
        let nav = f.try_to_nav(db).map(|r| r.call_site);
        let fn_name = f.name(db).as_str().to_owned();
        r.push(TestItem {
            id: format!("{prefix_id}::{fn_name}"),
            kind: TestItemKind::Function,
            label: fn_name,
            parent: Some(prefix_id.clone()),
            file: nav.as_ref().map(|n| n.file_id),
            text_range: nav.as_ref().map(|n| n.focus_or_full_range()),
            runnable: runnable_fn(&sema, f),
        });
    }
    r
}

pub(crate) fn discover_tests_in_crate_by_test_id(
    db: &RootDatabase,
    crate_test_id: &str,
) -> Vec<TestItem> {
    let Some(crate_id) = find_crate_by_id(db, crate_test_id) else {
        return vec![];
    };
    discover_tests_in_crate(db, crate_id)
}

pub(crate) fn discover_tests_in_file(db: &RootDatabase, file_id: FileId) -> Vec<TestItem> {
    let sema = Semantics::new(db);

    let Some(module) = sema.file_to_module_def(file_id) else { return vec![] };
    let Some((mut tests, id)) = find_module_id_and_test_parents(&sema, module) else {
        return vec![];
    };
    tests.extend(discover_tests_in_module(db, module, id, true));
    tests
}

fn find_module_id_and_test_parents(
    sema: &Semantics<'_, RootDatabase>,
    module: Module,
) -> Option<(Vec<TestItem>, String)> {
    let Some(parent) = module.parent(sema.db) else {
        let name = module.krate().display_name(sema.db)?.to_string();
        return Some((
            vec![TestItem {
                id: name.clone(),
                kind: TestItemKind::Crate(module.krate().into()),
                label: name.clone(),
                parent: None,
                file: None,
                text_range: None,
                runnable: None,
            }],
            name,
        ));
    };
    let (mut r, mut id) = find_module_id_and_test_parents(sema, parent)?;
    let parent = Some(id.clone());
    id += "::";
    let module_name = &module.name(sema.db);
    let module_name = module_name.as_ref().map(|n| n.as_str()).unwrap_or("[mod without name]");
    id += module_name;
    let nav = NavigationTarget::from_module_to_decl(sema.db, module).call_site;
    r.push(TestItem {
        id: id.clone(),
        kind: TestItemKind::Module,
        label: module_name.to_owned(),
        parent,
        file: Some(nav.file_id),
        text_range: Some(nav.focus_or_full_range()),
        runnable: None,
    });
    Some((r, id))
}

pub(crate) fn discover_tests_in_crate(
    db: &RootDatabase,
    crate_id: base_db::Crate,
) -> Vec<TestItem> {
    if !crate_id.data(db).origin.is_local() {
        return vec![];
    }
    let Some(crate_test_id) = &crate_id.extra_data(db).display_name else {
        return vec![];
    };
    let kind = TestItemKind::Crate(crate_id);
    let crate_test_id = crate_test_id.to_string();
    let crate_id: Crate = crate_id.into();
    let module = crate_id.root_module();
    let mut r = vec![TestItem {
        id: crate_test_id.clone(),
        kind,
        label: crate_test_id.clone(),
        parent: None,
        file: None,
        text_range: None,
        runnable: None,
    }];
    r.extend(discover_tests_in_module(db, module, crate_test_id, false));
    r
}
