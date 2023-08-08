use ide_db::{
    base_db::{CrateOrigin, FileId, SourceDatabase},
    FxIndexSet, RootDatabase,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateInfo {
    pub name: Option<String>,
    pub version: Option<String>,
    pub root_file_id: FileId,
}

// Feature: Show Dependency Tree
//
// Shows a view tree with all the dependencies of this project
//
// |===
// | Editor  | Panel Name
//
// | VS Code | **Rust Dependencies**
// |===
//
// image::https://user-images.githubusercontent.com/5748995/229394139-2625beab-f4c9-484b-84ed-ad5dee0b1e1a.png[]
pub(crate) fn fetch_crates(db: &RootDatabase) -> FxIndexSet<CrateInfo> {
    let crate_graph = db.crate_graph();
    crate_graph
        .iter()
        .map(|crate_id| &crate_graph[crate_id])
        .filter(|&data| !matches!(data.origin, CrateOrigin::Local { .. }))
        .map(|data| crate_info(data))
        .collect()
}

fn crate_info(data: &ide_db::base_db::CrateData) -> CrateInfo {
    let crate_name = crate_name(data);
    let version = data.version.clone();
    CrateInfo { name: crate_name, version, root_file_id: data.root_file_id }
}

fn crate_name(data: &ide_db::base_db::CrateData) -> Option<String> {
    data.display_name.as_ref().map(|it| it.canonical_name().to_owned())
}
