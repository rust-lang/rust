use std::collections::{HashMap, HashSet};

use crate::model::{
    ArtifactRecord, BuildScriptExecutedMessage, InvocationRecord, ProvenanceEdge, ProvenanceGraph,
    ProvenanceNode,
};

pub fn build_provenance(
    left_run_label: &str,
    right_run_label: &str,
    left_artifacts: &[ArtifactRecord],
    right_artifacts: &[ArtifactRecord],
    left_build_scripts: &[BuildScriptExecutedMessage],
    right_build_scripts: &[BuildScriptExecutedMessage],
    left_invocations: &HashMap<String, Vec<InvocationRecord>>,
    right_invocations: &HashMap<String, Vec<InvocationRecord>>,
) -> ProvenanceGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut node_ids = HashSet::<String>::new();

    let left_cmd = format!("cmd:{left_run_label}");
    let right_cmd = format!("cmd:{right_run_label}");
    add_node(
        &mut nodes,
        &mut node_ids,
        ProvenanceNode {
            id: left_cmd.clone(),
            kind: "CommandNode".to_string(),
            label: left_run_label.to_string(),
        },
    );
    add_node(
        &mut nodes,
        &mut node_ids,
        ProvenanceNode {
            id: right_cmd.clone(),
            kind: "CommandNode".to_string(),
            label: right_run_label.to_string(),
        },
    );

    add_run_invocations("left", &left_cmd, left_invocations, &mut nodes, &mut edges, &mut node_ids);
    add_run_invocations(
        "right",
        &right_cmd,
        right_invocations,
        &mut nodes,
        &mut edges,
        &mut node_ids,
    );

    add_build_script_messages("left", left_build_scripts, &mut nodes, &mut edges, &mut node_ids);
    add_build_script_messages("right", right_build_scripts, &mut nodes, &mut edges, &mut node_ids);

    add_run_artifacts("left", left_artifacts, &mut nodes, &mut edges, &mut node_ids);
    add_run_artifacts("right", right_artifacts, &mut nodes, &mut edges, &mut node_ids);

    ProvenanceGraph { nodes, edges }
}

fn add_run_invocations(
    side: &str,
    command_id: &str,
    invocations: &HashMap<String, Vec<InvocationRecord>>,
    nodes: &mut Vec<ProvenanceNode>,
    edges: &mut Vec<ProvenanceEdge>,
    node_ids: &mut HashSet<String>,
) {
    for (tool, invs) in invocations {
        for inv in invs {
            let inv_id = format!("inv:{side}:{}", inv.id);
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: inv_id.clone(),
                    kind: "InvocationNode".to_string(),
                    label: format!("{tool}:{}", inv.crate_name.clone().unwrap_or_default()),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: command_id.to_string(),
                    to: inv_id.clone(),
                    kind: "spawns".to_string(),
                },
            );

            for key in inv.env.keys() {
                let env_id = format!("env:{side}:{key}");
                add_node(
                    nodes,
                    node_ids,
                    ProvenanceNode {
                        id: env_id.clone(),
                        kind: "EnvVarNode".to_string(),
                        label: key.clone(),
                    },
                );
                add_edge(
                    edges,
                    ProvenanceEdge {
                        from: env_id,
                        to: inv_id.clone(),
                        kind: "reads-env".to_string(),
                    },
                );
            }

            if let Some(pkg) = &inv.package_id {
                let pkg_id = format!("pkg:{side}:{pkg}");
                add_node(
                    nodes,
                    node_ids,
                    ProvenanceNode {
                        id: pkg_id.clone(),
                        kind: "CargoPackageNode".to_string(),
                        label: pkg.clone(),
                    },
                );
                add_edge(
                    edges,
                    ProvenanceEdge {
                        from: pkg_id.clone(),
                        to: inv_id.clone(),
                        kind: "compiles".to_string(),
                    },
                );
                if let Some(target) = &inv.crate_name {
                    let target_id = target_node_id(side, pkg, target, &inv.crate_types);
                    add_node(
                        nodes,
                        node_ids,
                        ProvenanceNode {
                            id: target_id.clone(),
                            kind: "CargoTargetNode".to_string(),
                            label: target.clone(),
                        },
                    );
                    add_edge(
                        edges,
                        ProvenanceEdge {
                            from: pkg_id,
                            to: target_id.clone(),
                            kind: "depends-on".to_string(),
                        },
                    );
                    add_edge(
                        edges,
                        ProvenanceEdge {
                            from: target_id,
                            to: inv_id,
                            kind: "compiles".to_string(),
                        },
                    );
                }
            }
        }
    }
}

fn add_run_artifacts(
    side: &str,
    artifacts: &[ArtifactRecord],
    nodes: &mut Vec<ProvenanceNode>,
    edges: &mut Vec<ProvenanceEdge>,
    node_ids: &mut HashSet<String>,
) {
    for art in artifacts {
        let art_id = format!("art:{side}:{}", art.id);
        let art_kind = if art.kind == "out-dir-file" { "OutDirFileNode" } else { "ArtifactNode" };
        add_node(
            nodes,
            node_ids,
            ProvenanceNode {
                id: art_id.clone(),
                kind: art_kind.to_string(),
                label: art.rel_path.clone(),
            },
        );

        if let Some(inv) = &art.producer_invocation {
            add_edge(
                edges,
                ProvenanceEdge {
                    from: format!("inv:{side}:{inv}"),
                    to: art_id.clone(),
                    kind: "emits".to_string(),
                },
            );
        }

        if let Some(pkg) = &art.package_id {
            let pkg_id = format!("pkg:{side}:{pkg}");
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: pkg_id.clone(),
                    kind: "CargoPackageNode".to_string(),
                    label: pkg.clone(),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: pkg_id.clone(),
                    to: art_id.clone(),
                    kind: "generated-by".to_string(),
                },
            );

            if let Some(target_name) = &art.target_name {
                let target_id = target_node_id(side, pkg, target_name, &art.target_kind);
                add_node(
                    nodes,
                    node_ids,
                    ProvenanceNode {
                        id: target_id.clone(),
                        kind: "CargoTargetNode".to_string(),
                        label: target_name.clone(),
                    },
                );
                add_edge(
                    edges,
                    ProvenanceEdge {
                        from: target_id.clone(),
                        to: art_id.clone(),
                        kind: "emits".to_string(),
                    },
                );

                if art.target_kind.iter().any(|k| k == "proc-macro") {
                    let proc_id = format!("proc-macro:{side}:{pkg}:{target_name}");
                    add_node(
                        nodes,
                        node_ids,
                        ProvenanceNode {
                            id: proc_id.clone(),
                            kind: "ProcMacroNode".to_string(),
                            label: target_name.clone(),
                        },
                    );
                    add_edge(
                        edges,
                        ProvenanceEdge {
                            from: proc_id,
                            to: art_id.clone(),
                            kind: "generated-by".to_string(),
                        },
                    );
                }
            }

            if art.kind == "out-dir-file" || art.target_kind.iter().any(|k| k == "build-script") {
                let build_script_id = format!("build-script:{side}:{pkg}");
                add_node(
                    nodes,
                    node_ids,
                    ProvenanceNode {
                        id: build_script_id.clone(),
                        kind: "BuildScriptNode".to_string(),
                        label: pkg.clone(),
                    },
                );
                add_edge(
                    edges,
                    ProvenanceEdge {
                        from: build_script_id,
                        to: art_id.clone(),
                        kind: "generated-by".to_string(),
                    },
                );
            }
        }

        for input in &art.inputs {
            let src_id = format!("src:{side}:{input}");
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: src_id.clone(),
                    kind: "SourceFileNode".to_string(),
                    label: input.clone(),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: src_id,
                    to: art_id.clone(),
                    kind: "reads-source".to_string(),
                },
            );
        }
    }
}

fn add_build_script_messages(
    side: &str,
    messages: &[BuildScriptExecutedMessage],
    nodes: &mut Vec<ProvenanceNode>,
    edges: &mut Vec<ProvenanceEdge>,
    node_ids: &mut HashSet<String>,
) {
    for msg in messages {
        let build_script_id = format!("build-script:{side}:{}", msg.package_id);
        add_node(
            nodes,
            node_ids,
            ProvenanceNode {
                id: build_script_id.clone(),
                kind: "BuildScriptNode".to_string(),
                label: msg.package_id.clone(),
            },
        );

        for (k, v) in &msg.env {
            let env_id = format!("env:{side}:{k}");
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: env_id.clone(),
                    kind: "EnvVarNode".to_string(),
                    label: format!("{k}={v}"),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: build_script_id.clone(),
                    to: env_id,
                    kind: "sets-env".to_string(),
                },
            );
        }

        for lib in &msg.linked_libs {
            let link_id = format!("link:{side}:{}:lib:{lib}", msg.package_id);
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: link_id.clone(),
                    kind: "LinkArgNode".to_string(),
                    label: format!("lib:{lib}"),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: build_script_id.clone(),
                    to: link_id,
                    kind: "links-via".to_string(),
                },
            );
        }

        for path in &msg.linked_paths {
            let link_id = format!("link:{side}:{}:path:{path}", msg.package_id);
            add_node(
                nodes,
                node_ids,
                ProvenanceNode {
                    id: link_id.clone(),
                    kind: "LinkArgNode".to_string(),
                    label: format!("path:{path}"),
                },
            );
            add_edge(
                edges,
                ProvenanceEdge {
                    from: build_script_id.clone(),
                    to: link_id,
                    kind: "links-via".to_string(),
                },
            );
        }
    }
}

fn add_node(nodes: &mut Vec<ProvenanceNode>, seen: &mut HashSet<String>, node: ProvenanceNode) {
    if seen.insert(node.id.clone()) {
        nodes.push(node);
    }
}

fn add_edge(edges: &mut Vec<ProvenanceEdge>, edge: ProvenanceEdge) {
    edges.push(edge);
}

fn target_node_id(side: &str, pkg: &str, target_name: &str, kinds: &[String]) -> String {
    let suffix = if kinds.is_empty() { "unknown".to_string() } else { kinds.join(",") };
    format!("target:{side}:{pkg}:{target_name}:{suffix}")
}
