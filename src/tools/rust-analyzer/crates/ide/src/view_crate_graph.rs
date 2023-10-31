use dot::{Id, LabelText};
use ide_db::{
    base_db::{CrateGraph, CrateId, Dependency, SourceDatabase, SourceDatabaseExt},
    FxHashSet, RootDatabase,
};
use triomphe::Arc;

// Feature: View Crate Graph
//
// Renders the currently loaded crate graph as an SVG graphic. Requires the `dot` tool, which
// is part of graphviz, to be installed.
//
// Only workspace crates are included, no crates.io dependencies or sysroot crates.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: View Crate Graph**
// |===
pub(crate) fn view_crate_graph(db: &RootDatabase, full: bool) -> Result<String, String> {
    let crate_graph = db.crate_graph();
    let crates_to_render = crate_graph
        .iter()
        .filter(|krate| {
            if full {
                true
            } else {
                // Only render workspace crates
                let root_id = db.file_source_root(crate_graph[*krate].root_file_id);
                !db.source_root(root_id).is_library
            }
        })
        .collect();
    let graph = DotCrateGraph { graph: crate_graph, crates_to_render };

    let mut dot = Vec::new();
    dot::render(&graph, &mut dot).unwrap();
    Ok(String::from_utf8(dot).unwrap())
}

struct DotCrateGraph {
    graph: Arc<CrateGraph>,
    crates_to_render: FxHashSet<CrateId>,
}

type Edge<'a> = (CrateId, &'a Dependency);

impl<'a> dot::GraphWalk<'a, CrateId, Edge<'a>> for DotCrateGraph {
    fn nodes(&'a self) -> dot::Nodes<'a, CrateId> {
        self.crates_to_render.iter().copied().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.crates_to_render
            .iter()
            .flat_map(|krate| {
                self.graph[*krate]
                    .dependencies
                    .iter()
                    .filter(|dep| self.crates_to_render.contains(&dep.crate_id))
                    .map(move |dep| (*krate, dep))
            })
            .collect()
    }

    fn source(&'a self, edge: &Edge<'a>) -> CrateId {
        edge.0
    }

    fn target(&'a self, edge: &Edge<'a>) -> CrateId {
        edge.1.crate_id
    }
}

impl<'a> dot::Labeller<'a, CrateId, Edge<'a>> for DotCrateGraph {
    fn graph_id(&'a self) -> Id<'a> {
        Id::new("rust_analyzer_crate_graph").unwrap()
    }

    fn node_id(&'a self, n: &CrateId) -> Id<'a> {
        Id::new(format!("_{}", u32::from(n.into_raw()))).unwrap()
    }

    fn node_shape(&'a self, _node: &CrateId) -> Option<LabelText<'a>> {
        Some(LabelText::LabelStr("box".into()))
    }

    fn node_label(&'a self, n: &CrateId) -> LabelText<'a> {
        let name = self.graph[*n].display_name.as_ref().map_or("(unnamed crate)", |name| &*name);
        LabelText::LabelStr(name.into())
    }
}
