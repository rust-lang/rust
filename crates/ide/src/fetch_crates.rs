use ide_db::{
    base_db::{CrateOrigin, SourceDatabase, SourceDatabaseExt},
    RootDatabase,
};

#[derive(Debug)]
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
pub(crate) fn fetch_crates(db: &RootDatabase) -> Vec<CrateInfo> {
    let crate_graph = db.crate_graph();
    crate_graph
        .iter()
        .map(|crate_id| &crate_graph[crate_id])
        .filter(|&data| !matches!(data.origin, CrateOrigin::Local { .. }))
        .map(|data| {
            let crate_name = crate_name(data);
            let version = data.version.clone().unwrap_or_else(|| "".to_owned());
            let crate_path = crate_path(db, data, &crate_name);

            CrateInfo { name: crate_name, version, path: crate_path }
        })
        .collect()
}

fn crate_name(data: &ide_db::base_db::CrateData) -> String {
    data.display_name
        .clone()
        .map(|it| it.canonical_name().to_owned())
        .unwrap_or("unknown".to_string())
}

fn crate_path(db: &RootDatabase, data: &ide_db::base_db::CrateData, crate_name: &str) -> String {
    let source_root_id = db.file_source_root(data.root_file_id);
    let source_root = db.source_root(source_root_id);
    let source_root_path = source_root.path_for_file(&data.root_file_id);
    match source_root_path.cloned() {
        Some(mut root_path) => {
            let mut crate_path = "".to_string();
            while let Some(vfs_path) = root_path.parent() {
                match vfs_path.name_and_extension() {
                    Some((name, _)) => {
                        if name.starts_with(crate_name) {
                            crate_path = vfs_path.to_string();
                            break;
                        }
                    }
                    None => break,
                }
                root_path = vfs_path;
            }
            crate_path
        }
        None => "".to_owned(),
    }
}
