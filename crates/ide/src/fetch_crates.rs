use ide_db::{
    base_db::{CrateOrigin, SourceDatabase},
    FxIndexSet, RootDatabase,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateInfo {
    pub name: String,
    pub version: String,
    pub path: String,
}

// Feature: Show Dependency Tree
//
// Shows a view tree with all the dependencies of this project
//
// |===
// image::https://user-images.githubusercontent.com/5748995/229394139-2625beab-f4c9-484b-84ed-ad5dee0b1e1a.png[]
pub(crate) fn fetch_crates(db: &RootDatabase) -> FxIndexSet<CrateInfo> {
    let crate_graph = db.crate_graph();
    crate_graph
        .iter()
        .map(|crate_id| &crate_graph[crate_id])
        .filter(|&data| !matches!(data.origin, CrateOrigin::Local { .. }))
        .filter_map(|data| crate_info(data))
        .collect()
}

fn crate_info(data: &ide_db::base_db::CrateData) -> Option<CrateInfo> {
    let crate_name = crate_name(data);
    let crate_path = data.crate_root_path.as_ref().map(|p| p.display().to_string());
    if let Some(crate_path) = crate_path {
        let version = data.version.clone().unwrap_or_else(|| "".to_owned());
        Some(CrateInfo { name: crate_name, version, path: crate_path })
    } else {
        None
    }
}

fn crate_name(data: &ide_db::base_db::CrateData) -> String {
    data.display_name
        .clone()
        .map(|it| it.canonical_name().to_owned())
        .unwrap_or("unknown".to_string())
}
