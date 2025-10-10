use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::io::BufWriter;
use std::path::Path;

use crate::core::builder::{AnyDebug, Step, pretty_step_name};
use crate::t;

/// Records the executed steps and their dependencies in a directed graph,
/// which can then be rendered into a DOT file for visualization.
///
/// The graph visualizes the first execution of a step with a solid edge,
/// and cached executions of steps with a dashed edge.
/// If you only want to see first executions, you can modify the code in `DotGraph` to
/// always set `cached: false`.
#[derive(Default)]
pub struct StepGraph {
    /// We essentially store one graph per dry run mode.
    graphs: HashMap<String, DotGraph>,
}

impl StepGraph {
    pub fn register_step_execution<S: Step>(
        &mut self,
        step: &S,
        parent: Option<&Box<dyn AnyDebug>>,
        dry_run: bool,
    ) {
        let key = get_graph_key(dry_run);
        let graph = self.graphs.entry(key.to_string()).or_insert_with(|| DotGraph::default());

        // The debug output of the step sort of serves as the unique identifier of it.
        // We use it to access the node ID of parents to generate edges.
        // We could probably also use addresses on the heap from the `Box`, but this seems less
        // magical.
        let node_key = render_step(step);

        let label = if let Some(metadata) = step.metadata() {
            format!(
                "{}{} [{}]",
                metadata.get_name(),
                metadata.get_stage().map(|s| format!(" stage {s}")).unwrap_or_default(),
                metadata.get_target()
            )
        } else {
            pretty_step_name::<S>()
        };

        let node = Node { label, tooltip: node_key.clone() };
        let node_handle = graph.add_node(node_key, node);

        if let Some(parent) = parent {
            let parent_key = render_step(parent);
            if let Some(src_node_handle) = graph.get_handle_by_key(&parent_key) {
                graph.add_edge(src_node_handle, node_handle);
            }
        }
    }

    pub fn register_cached_step<S: Step>(
        &mut self,
        step: &S,
        parent: &Box<dyn AnyDebug>,
        dry_run: bool,
    ) {
        let key = get_graph_key(dry_run);
        let graph = self.graphs.get_mut(key).unwrap();

        let node_key = render_step(step);
        let parent_key = render_step(parent);

        if let Some(src_node_handle) = graph.get_handle_by_key(&parent_key) {
            if let Some(dst_node_handle) = graph.get_handle_by_key(&node_key) {
                graph.add_cached_edge(src_node_handle, dst_node_handle);
            }
        }
    }

    pub fn store_to_dot_files(self, directory: &Path) {
        for (key, graph) in self.graphs.into_iter() {
            let filename = directory.join(format!("step-graph{key}.dot"));
            t!(graph.render(&filename));
        }
    }
}

fn get_graph_key(dry_run: bool) -> &'static str {
    if dry_run { ".dryrun" } else { "" }
}

struct Node {
    label: String,
    tooltip: String,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct NodeHandle(usize);

/// Represents a dependency between two bootstrap steps.
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Edge {
    src: NodeHandle,
    dst: NodeHandle,
    // Was the corresponding execution of a step cached, or was the step actually executed?
    cached: bool,
}

// We could use a library for this, but they either:
// - require lifetimes, which gets annoying (dot_writer)
// - don't support tooltips (dot_graph)
// - have a lot of dependencies (graphviz_rust)
// - only have SVG export (layout-rs)
// - use a builder pattern that is very annoying to use here (tabbycat)
#[derive(Default)]
struct DotGraph {
    nodes: Vec<Node>,
    /// The `NodeHandle` represents an index within `self.nodes`
    edges: HashSet<Edge>,
    key_to_index: HashMap<String, NodeHandle>,
}

impl DotGraph {
    fn add_node(&mut self, key: String, node: Node) -> NodeHandle {
        let handle = NodeHandle(self.nodes.len());
        self.nodes.push(node);
        self.key_to_index.insert(key, handle);
        handle
    }

    fn add_edge(&mut self, src: NodeHandle, dst: NodeHandle) {
        self.edges.insert(Edge { src, dst, cached: false });
    }

    fn add_cached_edge(&mut self, src: NodeHandle, dst: NodeHandle) {
        // There's no point in rendering both cached and uncached edge
        let uncached = Edge { src, dst, cached: false };
        if !self.edges.contains(&uncached) {
            self.edges.insert(Edge { src, dst, cached: true });
        }
    }

    fn get_handle_by_key(&self, key: &str) -> Option<NodeHandle> {
        self.key_to_index.get(key).copied()
    }

    fn render(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = BufWriter::new(std::fs::File::create(path)?);
        writeln!(file, "digraph bootstrap_steps {{")?;
        for (index, node) in self.nodes.iter().enumerate() {
            writeln!(
                file,
                r#"{index} [label="{}", tooltip="{}"]"#,
                escape(&node.label),
                escape(&node.tooltip)
            )?;
        }

        let mut edges: Vec<&Edge> = self.edges.iter().collect();
        edges.sort();
        for edge in edges {
            let style = if edge.cached { "dashed" } else { "solid" };
            writeln!(file, r#"{} -> {} [style="{style}"]"#, edge.src.0, edge.dst.0)?;
        }

        writeln!(file, "}}")
    }
}

fn render_step(step: &dyn Debug) -> String {
    format!("{step:?}")
}

/// Normalizes the string so that it can be rendered into a DOT file.
fn escape(input: &str) -> String {
    input.replace("\"", "\\\"")
}
