use std::{
    error::Error,
    io::{Read, Write},
    process::{Command, Stdio},
    sync::Arc,
};

use dot::Id;
use ide_db::{
    base_db::{CrateGraph, CrateId, Dependency, SourceDatabase},
    RootDatabase,
};

// Feature: View Crate Graph
//
// Renders the currently loaded crate graph as an SVG graphic. Requires the `dot` tool to be
// installed.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: View Crate Graph**
// |===
pub(crate) fn view_crate_graph(db: &RootDatabase) -> Result<String, String> {
    let mut dot = Vec::new();
    let graph = DotCrateGraph(db.crate_graph());
    dot::render(&graph, &mut dot).unwrap();

    render_svg(&dot).map_err(|e| e.to_string())
}

fn render_svg(dot: &[u8]) -> Result<String, Box<dyn Error>> {
    // We shell out to `dot` to render to SVG, as there does not seem to be a pure-Rust renderer.
    let child = Command::new("dot")
        .arg("-Tsvg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to spawn `dot -Tsvg`: {}", err))?;
    child.stdin.unwrap().write_all(&dot)?;

    let mut svg = String::new();
    child.stdout.unwrap().read_to_string(&mut svg)?;
    Ok(svg)
}

struct DotCrateGraph(Arc<CrateGraph>);

type Edge<'a> = (CrateId, &'a Dependency);

impl<'a> dot::GraphWalk<'a, CrateId, Edge<'a>> for DotCrateGraph {
    fn nodes(&'a self) -> dot::Nodes<'a, CrateId> {
        self.0.iter().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.0
            .iter()
            .flat_map(|krate| self.0[krate].dependencies.iter().map(move |dep| (krate, dep)))
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
        let name = self.0[*n].display_name.as_ref().map_or("_missing_name_", |name| &*name);
        Id::new(format!("{}_{}", name, n.0)).unwrap()
    }
}
