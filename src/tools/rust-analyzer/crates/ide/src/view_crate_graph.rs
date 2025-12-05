use dot::{Id, LabelText};
use ide_db::base_db::salsa::plumbing::AsId;
use ide_db::{
    FxHashMap, RootDatabase,
    base_db::{
        BuiltCrateData, BuiltDependency, Crate, ExtraCrateData, RootQueryDb, SourceDatabase,
    },
};

// Feature: View Crate Graph
//
// Renders the currently loaded crate graph as an SVG graphic. Requires the `dot` tool, which
// is part of graphviz, to be installed.
//
// Only workspace crates are included, no crates.io dependencies or sysroot crates.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: View Crate Graph** |
pub(crate) fn view_crate_graph(db: &RootDatabase, full: bool) -> Result<String, String> {
    let all_crates = db.all_crates();
    let crates_to_render = all_crates
        .iter()
        .copied()
        .map(|krate| (krate, (krate.data(db), krate.extra_data(db))))
        .filter(|(_, (crate_data, _))| {
            if full {
                true
            } else {
                // Only render workspace crates
                let root_id = db.file_source_root(crate_data.root_file_id).source_root_id(db);
                !db.source_root(root_id).source_root(db).is_library
            }
        })
        .collect();
    let graph = DotCrateGraph { crates_to_render };

    let mut dot = Vec::new();
    dot::render(&graph, &mut dot).unwrap();
    Ok(String::from_utf8(dot).unwrap())
}

struct DotCrateGraph<'db> {
    crates_to_render: FxHashMap<Crate, (&'db BuiltCrateData, &'db ExtraCrateData)>,
}

type Edge<'a> = (Crate, &'a BuiltDependency);

impl<'a> dot::GraphWalk<'a, Crate, Edge<'a>> for DotCrateGraph<'_> {
    fn nodes(&'a self) -> dot::Nodes<'a, Crate> {
        self.crates_to_render.keys().copied().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.crates_to_render
            .iter()
            .flat_map(|(krate, (crate_data, _))| {
                crate_data
                    .dependencies
                    .iter()
                    .filter(|dep| self.crates_to_render.contains_key(&dep.crate_id))
                    .map(move |dep| (*krate, dep))
            })
            .collect()
    }

    fn source(&'a self, edge: &Edge<'a>) -> Crate {
        edge.0
    }

    fn target(&'a self, edge: &Edge<'a>) -> Crate {
        edge.1.crate_id
    }
}

impl<'a> dot::Labeller<'a, Crate, Edge<'a>> for DotCrateGraph<'_> {
    fn graph_id(&'a self) -> Id<'a> {
        Id::new("rust_analyzer_crate_graph").unwrap()
    }

    fn node_id(&'a self, n: &Crate) -> Id<'a> {
        let id = n.as_id().index();
        Id::new(format!("_{id:?}")).unwrap()
    }

    fn node_shape(&'a self, _node: &Crate) -> Option<LabelText<'a>> {
        Some(LabelText::LabelStr("box".into()))
    }

    fn node_label(&'a self, n: &Crate) -> LabelText<'a> {
        let name = self.crates_to_render[n]
            .1
            .display_name
            .as_ref()
            .map_or("(unnamed crate)", |name| name.as_str());
        LabelText::LabelStr(name.into())
    }
}
