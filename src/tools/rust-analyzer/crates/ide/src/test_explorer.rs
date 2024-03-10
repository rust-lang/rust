//! Discovers tests

use hir::{Crate, Module, ModuleDef, Semantics};
use ide_db::{
    base_db::{CrateGraph, CrateId, FileId, SourceDatabase},
    RootDatabase,
};
use syntax::TextRange;

use crate::{navigation_target::ToNav, runnables::runnable_fn, Runnable, TryToNav};

#[derive(Debug)]
pub enum TestItemKind {
    Crate,
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
    let crate_graph = db.crate_graph();
    crate_graph
        .iter()
        .filter(|&id| crate_graph[id].origin.is_local())
        .filter_map(|id| Some(crate_graph[id].display_name.as_ref()?.to_string()))
        .map(|id| TestItem {
            kind: TestItemKind::Crate,
            label: id.clone(),
            id,
            parent: None,
            file: None,
            text_range: None,
            runnable: None,
        })
        .collect()
}

fn find_crate_by_id(crate_graph: &CrateGraph, crate_id: &str) -> Option<CrateId> {
    // here, we use display_name as the crate id. This is not super ideal, but it works since we
    // only show tests for the local crates.
    crate_graph.iter().find(|&id| {
        crate_graph[id].origin.is_local()
            && crate_graph[id].display_name.as_ref().is_some_and(|x| x.to_string() == crate_id)
    })
}

fn discover_tests_in_module(db: &RootDatabase, module: Module, prefix_id: String) -> Vec<TestItem> {
    let sema = Semantics::new(db);

    let mut r = vec![];
    for c in module.children(db) {
        let module_name =
            c.name(db).as_ref().and_then(|n| n.as_str()).unwrap_or("[mod without name]").to_owned();
        let module_id = format!("{prefix_id}::{module_name}");
        let module_children = discover_tests_in_module(db, c, module_id.clone());
        if !module_children.is_empty() {
            let nav = c.to_nav(db).call_site;
            r.push(TestItem {
                id: module_id,
                kind: TestItemKind::Module,
                label: module_name,
                parent: Some(prefix_id.clone()),
                file: Some(nav.file_id),
                text_range: Some(nav.focus_or_full_range()),
                runnable: None,
            });
            r.extend(module_children);
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
        let fn_name = f.name(db).as_str().unwrap_or("[function without name]").to_owned();
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
    let crate_graph = db.crate_graph();
    let Some(crate_id) = find_crate_by_id(&crate_graph, crate_test_id) else {
        return vec![];
    };
    discover_tests_in_crate(db, crate_id)
}

pub(crate) fn discover_tests_in_crate(db: &RootDatabase, crate_id: CrateId) -> Vec<TestItem> {
    let crate_graph = db.crate_graph();
    if !crate_graph[crate_id].origin.is_local() {
        return vec![];
    }
    let Some(crate_test_id) = &crate_graph[crate_id].display_name else {
        return vec![];
    };
    let crate_test_id = crate_test_id.to_string();
    let crate_id: Crate = crate_id.into();
    let module = crate_id.root_module();
    let mut r = vec![TestItem {
        id: crate_test_id.clone(),
        kind: TestItemKind::Crate,
        label: crate_test_id.clone(),
        parent: None,
        file: None,
        text_range: None,
        runnable: None,
    }];
    r.extend(discover_tests_in_module(db, module, crate_test_id));
    r
}
