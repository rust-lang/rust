use ide_db::{
    base_db::{CrateOrigin, SourceDatabase, SourceDatabaseExt},
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
        .filter_map(|data| crate_info(data, db))
        .collect()
}

fn crate_info(data: &ide_db::base_db::CrateData, db: &RootDatabase) -> Option<CrateInfo> {
    let crate_name = crate_name(data);
    let crate_path = crate_path(db, data, &crate_name);
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

fn crate_path(
    db: &RootDatabase,
    data: &ide_db::base_db::CrateData,
    crate_name: &str,
) -> Option<String> {
    let source_root_id = db.file_source_root(data.root_file_id);
    let source_root = db.source_root(source_root_id);
    let source_root_path = source_root.path_for_file(&data.root_file_id);
    source_root_path.cloned().and_then(|mut root_path| {
        let mut crate_path = None;
        while let Some(vfs_path) = root_path.parent() {
            match vfs_path.name_and_extension() {
                Some((name, _)) => {
                    if name.starts_with(crate_name) {
                        crate_path = Some(vfs_path.to_string());
                        break;
                    }
                }
                None => break,
            }
            root_path = vfs_path;
        }
        crate_path
    })
}
