//! This module is responsible for implementing handlers for Language Server
//! Protocol. This module specifically handles requests.

use std::{fs, io::Write as _, ops::Not, process::Stdio};

use anyhow::Context;

use base64::{Engine, prelude::BASE64_STANDARD};
use ide::{
    AnnotationConfig, AssistKind, AssistResolveStrategy, Cancellable, CompletionFieldsToResolve,
    FilePosition, FileRange, HoverAction, HoverGotoTypeData, InlayFieldsToResolve, Query,
    RangeInfo, ReferenceCategory, Runnable, RunnableKind, SingleResolve, SourceChange, TextEdit,
};
use ide_db::{FxHashMap, SymbolKind};
use itertools::Itertools;
use lsp_server::ErrorCode;
use lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    CodeLens, CompletionItem, FoldingRange, FoldingRangeParams, HoverContents, InlayHint,
    InlayHintParams, Location, LocationLink, Position, PrepareRenameResponse, Range, RenameParams,
    ResourceOp, ResourceOperationKind, SemanticTokensDeltaParams, SemanticTokensFullDeltaResult,
    SemanticTokensParams, SemanticTokensRangeParams, SemanticTokensRangeResult,
    SemanticTokensResult, SymbolInformation, SymbolTag, TextDocumentIdentifier, Url, WorkspaceEdit,
};
use paths::Utf8PathBuf;
use project_model::{CargoWorkspace, ManifestPath, ProjectWorkspaceKind, TargetKind};
use serde_json::json;
use stdx::{format_to, never};
use syntax::{TextRange, TextSize};
use triomphe::Arc;
use vfs::{AbsPath, AbsPathBuf, FileId, VfsPath};

use crate::{
    config::{Config, RustfmtConfig, WorkspaceSymbolConfig},
    diagnostics::convert_diagnostic,
    global_state::{FetchWorkspaceRequest, GlobalState, GlobalStateSnapshot},
    line_index::LineEndings,
    lsp::{
        LspError, completion_item_hash,
        ext::{
            InternalTestingFetchConfigOption, InternalTestingFetchConfigParams,
            InternalTestingFetchConfigResponse,
        },
        from_proto, to_proto,
        utils::{all_edits_are_disjoint, invalid_params_error},
    },
    lsp_ext::{
        self, CrateInfoResult, ExternalDocsPair, ExternalDocsResponse, FetchDependencyListParams,
        FetchDependencyListResult, PositionOrRange, ViewCrateGraphParams, WorkspaceSymbolParams,
    },
    target_spec::{CargoTargetSpec, TargetSpec},
    test_runner::{CargoTestHandle, TestTarget},
    try_default,
};

pub(crate) fn handle_workspace_reload(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    state.proc_macro_clients = Arc::from_iter([]);
    state.build_deps_changed = false;

    let req = FetchWorkspaceRequest { path: None, force_crate_graph_reload: false };
    state.fetch_workspaces_queue.request_op("reload workspace request".to_owned(), req);
    Ok(())
}

pub(crate) fn handle_proc_macros_rebuild(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    state.proc_macro_clients = Arc::from_iter([]);
    state.build_deps_changed = false;

    state.fetch_build_data_queue.request_op("rebuild proc macros request".to_owned(), ());
    Ok(())
}

pub(crate) fn handle_analyzer_status(
    snap: GlobalStateSnapshot,
    params: lsp_ext::AnalyzerStatusParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_analyzer_status").entered();

    let mut buf = String::new();

    let mut file_id = None;
    if let Some(tdi) = params.text_document {
        match from_proto::file_id(&snap, &tdi.uri) {
            Ok(Some(it)) => file_id = Some(it),
            Ok(None) => {}
            Err(_) => format_to!(buf, "file {} not found in vfs", tdi.uri),
        }
    }

    if snap.workspaces.is_empty() {
        buf.push_str("No workspaces\n")
    } else {
        buf.push_str("Workspaces:\n");
        format_to!(
            buf,
            "Loaded {:?} packages across {} workspace{}.\n",
            snap.workspaces.iter().map(|w| w.n_packages()).sum::<usize>(),
            snap.workspaces.len(),
            if snap.workspaces.len() == 1 { "" } else { "s" }
        );

        format_to!(
            buf,
            "Workspace root folders: {:?}",
            snap.workspaces.iter().map(|ws| ws.manifest_or_root()).collect::<Vec<&AbsPath>>()
        );
    }
    buf.push_str("\nAnalysis:\n");
    buf.push_str(
        &snap
            .analysis
            .status(file_id)
            .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
    );

    buf.push_str("\nVersion: \n");
    format_to!(buf, "{}", crate::version());

    buf.push_str("\nConfiguration: \n");
    format_to!(buf, "{:#?}", snap.config);

    Ok(buf)
}

pub(crate) fn handle_memory_usage(state: &mut GlobalState, _: ()) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_memory_usage").entered();
    let mem = state.analysis_host.per_query_memory_usage();

    let mut out = String::new();
    for (name, bytes, entries) in mem {
        format_to!(out, "{:>8} {:>6} {}\n", bytes, entries, name);
    }
    format_to!(out, "{:>8}        Remaining\n", profile::memory_usage().allocated);

    Ok(out)
}

pub(crate) fn handle_view_syntax_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ViewSyntaxTreeParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_view_syntax_tree").entered();
    let id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let res = snap.analysis.view_syntax_tree(id)?;
    Ok(res)
}

pub(crate) fn handle_view_hir(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_view_hir").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let res = snap.analysis.view_hir(position)?;
    Ok(res)
}

pub(crate) fn handle_view_mir(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_view_mir").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let res = snap.analysis.view_mir(position)?;
    Ok(res)
}

pub(crate) fn handle_interpret_function(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_interpret_function").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let res = snap.analysis.interpret_function(position)?;
    Ok(res)
}

pub(crate) fn handle_view_file_text(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentIdentifier,
) -> anyhow::Result<String> {
    let file_id = try_default!(from_proto::file_id(&snap, &params.uri)?);
    Ok(snap.analysis.file_text(file_id)?.to_string())
}

pub(crate) fn handle_view_item_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ViewItemTreeParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_view_item_tree").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let res = snap.analysis.view_item_tree(file_id)?;
    Ok(res)
}

// cargo test requires:
// - the package is a member of the workspace
// - the target in the package is not a build script (custom-build)
// - the package name - the root of the test identifier supplied to this handler can be
//   a package or a target inside a package.
// - the target name - if the test identifier is a target, it's needed in addition to the
//   package name to run the right test
// - real names - the test identifier uses the namespace form where hyphens are replaced with
//   underscores. cargo test requires the real name.
// - the target kind e.g. bin or lib
fn all_test_targets(cargo: &CargoWorkspace) -> impl Iterator<Item = TestTarget> {
    cargo.packages().filter(|p| cargo[*p].is_member).flat_map(|p| {
        let package = &cargo[p];
        package.targets.iter().filter_map(|t| {
            let target = &cargo[*t];
            if target.kind == TargetKind::BuildScript {
                None
            } else {
                Some(TestTarget {
                    package: package.name.clone(),
                    target: target.name.clone(),
                    kind: target.kind,
                })
            }
        })
    })
}

fn find_test_target(namespace_root: &str, cargo: &CargoWorkspace) -> Option<TestTarget> {
    all_test_targets(cargo).find(|t| namespace_root == t.target.replace('-', "_"))
}

pub(crate) fn handle_run_test(
    state: &mut GlobalState,
    params: lsp_ext::RunTestParams,
) -> anyhow::Result<()> {
    if let Some(_session) = state.test_run_session.take() {
        state.send_notification::<lsp_ext::EndRunTest>(());
    }

    let mut handles = vec![];
    for ws in &*state.workspaces {
        if let ProjectWorkspaceKind::Cargo { cargo, .. } = &ws.kind {
            // need to deduplicate `include` to avoid redundant test runs
            let tests = match params.include {
                Some(ref include) => include
                    .iter()
                    .unique()
                    .filter_map(|test| {
                        let (root, remainder) = match test.split_once("::") {
                            Some((root, remainder)) => (root.to_owned(), Some(remainder)),
                            None => (test.clone(), None),
                        };
                        if let Some(target) = find_test_target(&root, cargo) {
                            Some((target, remainder))
                        } else {
                            tracing::error!("Test target not found for: {test}");
                            None
                        }
                    })
                    .collect_vec(),
                None => all_test_targets(cargo).map(|target| (target, None)).collect(),
            };

            for (target, path) in tests {
                let handle = CargoTestHandle::new(
                    path,
                    state.config.cargo_test_options(None),
                    cargo.workspace_root(),
                    target,
                    state.test_run_sender.clone(),
                )?;
                handles.push(handle);
            }
        }
    }
    // Each process send finished signal twice, once for stdout and once for stderr
    state.test_run_remaining_jobs = 2 * handles.len();
    state.test_run_session = Some(handles);
    Ok(())
}

pub(crate) fn handle_discover_test(
    snap: GlobalStateSnapshot,
    params: lsp_ext::DiscoverTestParams,
) -> anyhow::Result<lsp_ext::DiscoverTestResults> {
    let _p = tracing::info_span!("handle_discover_test").entered();
    let (tests, scope) = match params.test_id {
        Some(id) => {
            let crate_id = id.split_once("::").map(|it| it.0).unwrap_or(&id);
            (
                snap.analysis.discover_tests_in_crate_by_test_id(crate_id)?,
                Some(vec![crate_id.to_owned()]),
            )
        }
        None => (snap.analysis.discover_test_roots()?, None),
    };

    Ok(lsp_ext::DiscoverTestResults {
        tests: tests
            .into_iter()
            .filter_map(|t| {
                let line_index = t.file.and_then(|f| snap.file_line_index(f).ok());
                to_proto::test_item(&snap, t, line_index.as_ref())
            })
            .collect(),
        scope,
        scope_file: None,
    })
}

pub(crate) fn handle_view_crate_graph(
    snap: GlobalStateSnapshot,
    params: ViewCrateGraphParams,
) -> anyhow::Result<String> {
    let _p = tracing::info_span!("handle_view_crate_graph").entered();
    let dot = snap.analysis.view_crate_graph(params.full)?.map_err(anyhow::Error::msg)?;
    Ok(dot)
}

pub(crate) fn handle_expand_macro(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ExpandMacroParams,
) -> anyhow::Result<Option<lsp_ext::ExpandedMacro>> {
    let _p = tracing::info_span!("handle_expand_macro").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, params.position)?;

    let res = snap.analysis.expand_macro(FilePosition { file_id, offset })?;
    Ok(res.map(|it| lsp_ext::ExpandedMacro { name: it.name, expansion: it.expansion }))
}

pub(crate) fn handle_selection_range(
    snap: GlobalStateSnapshot,
    params: lsp_types::SelectionRangeParams,
) -> anyhow::Result<Option<Vec<lsp_types::SelectionRange>>> {
    let _p = tracing::info_span!("handle_selection_range").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    let res: anyhow::Result<Vec<lsp_types::SelectionRange>> = params
        .positions
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position)?;
            let mut ranges = Vec::new();
            {
                let mut range = TextRange::new(offset, offset);
                loop {
                    ranges.push(range);
                    let frange = FileRange { file_id, range };
                    let next = snap.analysis.extend_selection(frange)?;
                    if next == range {
                        break;
                    } else {
                        range = next
                    }
                }
            }
            let mut range = lsp_types::SelectionRange {
                range: to_proto::range(&line_index, *ranges.last().unwrap()),
                parent: None,
            };
            for &r in ranges.iter().rev().skip(1) {
                range = lsp_types::SelectionRange {
                    range: to_proto::range(&line_index, r),
                    parent: Some(Box::new(range)),
                }
            }
            Ok(range)
        })
        .collect();

    Ok(Some(res?))
}

pub(crate) fn handle_matching_brace(
    snap: GlobalStateSnapshot,
    params: lsp_ext::MatchingBraceParams,
) -> anyhow::Result<Vec<Position>> {
    let _p = tracing::info_span!("handle_matching_brace").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    params
        .positions
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position);
            offset.map(|offset| {
                let offset = match snap.analysis.matching_brace(FilePosition { file_id, offset }) {
                    Ok(Some(matching_brace_offset)) => matching_brace_offset,
                    Err(_) | Ok(None) => offset,
                };
                to_proto::position(&line_index, offset)
            })
        })
        .collect()
}

pub(crate) fn handle_join_lines(
    snap: GlobalStateSnapshot,
    params: lsp_ext::JoinLinesParams,
) -> anyhow::Result<Vec<lsp_types::TextEdit>> {
    let _p = tracing::info_span!("handle_join_lines").entered();

    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let config = snap.config.join_lines();
    let line_index = snap.file_line_index(file_id)?;

    let mut res = TextEdit::default();
    for range in params.ranges {
        let range = from_proto::text_range(&line_index, range)?;
        let edit = snap.analysis.join_lines(&config, FileRange { file_id, range })?;
        match res.union(edit) {
            Ok(()) => (),
            Err(_edit) => {
                // just ignore overlapping edits
            }
        }
    }

    Ok(to_proto::text_edit_vec(&line_index, res))
}

pub(crate) fn handle_on_enter(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<Vec<lsp_ext::SnippetTextEdit>>> {
    let _p = tracing::info_span!("handle_on_enter").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let edit = match snap.analysis.on_enter(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let line_index = snap.file_line_index(position.file_id)?;
    let edit = to_proto::snippet_text_edit_vec(
        &line_index,
        true,
        edit,
        snap.config.change_annotation_support(),
    );
    Ok(Some(edit))
}

pub(crate) fn handle_on_type_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentOnTypeFormattingParams,
) -> anyhow::Result<Option<Vec<lsp_ext::SnippetTextEdit>>> {
    let _p = tracing::info_span!("handle_on_type_formatting").entered();
    let char_typed = params.ch.chars().next().unwrap_or('\0');
    if !snap.config.typing_trigger_chars().contains(char_typed) {
        return Ok(None);
    }

    let mut position =
        try_default!(from_proto::file_position(&snap, params.text_document_position)?);
    let line_index = snap.file_line_index(position.file_id)?;

    // in `ide`, the `on_type` invariant is that
    // `text.char_at(position) == typed_char`.
    position.offset -= TextSize::of('.');

    let text = snap.analysis.file_text(position.file_id)?;
    if stdx::never!(!text[usize::from(position.offset)..].starts_with(char_typed)) {
        return Ok(None);
    }

    let edit = snap.analysis.on_char_typed(position, char_typed)?;
    let edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let (_, (text_edit, snippet_edit)) = edit.source_file_edits.into_iter().next().unwrap();
    stdx::always!(snippet_edit.is_none(), "on type formatting shouldn't use structured snippets");

    let change = to_proto::snippet_text_edit_vec(
        &line_index,
        edit.is_snippet,
        text_edit,
        snap.config.change_annotation_support(),
    );
    Ok(Some(change))
}

pub(crate) fn empty_diagnostic_report() -> lsp_types::DocumentDiagnosticReportResult {
    lsp_types::DocumentDiagnosticReportResult::Report(lsp_types::DocumentDiagnosticReport::Full(
        lsp_types::RelatedFullDocumentDiagnosticReport {
            related_documents: None,
            full_document_diagnostic_report: lsp_types::FullDocumentDiagnosticReport {
                result_id: Some("rust-analyzer".to_owned()),
                items: vec![],
            },
        },
    ))
}

pub(crate) fn handle_document_diagnostics(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentDiagnosticParams,
) -> anyhow::Result<lsp_types::DocumentDiagnosticReportResult> {
    let file_id = match from_proto::file_id(&snap, &params.text_document.uri)? {
        Some(it) => it,
        None => return Ok(empty_diagnostic_report()),
    };
    let source_root = snap.analysis.source_root_id(file_id)?;
    if !snap.analysis.is_local_source_root(source_root)? {
        return Ok(empty_diagnostic_report());
    }
    let source_root = snap.analysis.source_root_id(file_id)?;
    let config = snap.config.diagnostics(Some(source_root));
    if !config.enabled {
        return Ok(empty_diagnostic_report());
    }
    let line_index = snap.file_line_index(file_id)?;
    let supports_related = snap.config.text_document_diagnostic_related_document_support();

    let mut related_documents = FxHashMap::default();
    let diagnostics = snap
        .analysis
        .full_diagnostics(&config, AssistResolveStrategy::None, file_id)?
        .into_iter()
        .filter_map(|d| {
            let file = d.range.file_id;
            if file == file_id {
                let diagnostic = convert_diagnostic(&line_index, d);
                return Some(diagnostic);
            }
            if supports_related {
                let (diagnostics, line_index) = related_documents
                    .entry(file)
                    .or_insert_with(|| (Vec::new(), snap.file_line_index(file).ok()));
                let diagnostic = convert_diagnostic(line_index.as_mut()?, d);
                diagnostics.push(diagnostic);
            }
            None
        });
    Ok(lsp_types::DocumentDiagnosticReportResult::Report(
        lsp_types::DocumentDiagnosticReport::Full(lsp_types::RelatedFullDocumentDiagnosticReport {
            full_document_diagnostic_report: lsp_types::FullDocumentDiagnosticReport {
                result_id: Some("rust-analyzer".to_owned()),
                items: diagnostics.collect(),
            },
            related_documents: related_documents.is_empty().not().then(|| {
                related_documents
                    .into_iter()
                    .map(|(id, (items, _))| {
                        (
                            to_proto::url(&snap, id),
                            lsp_types::DocumentDiagnosticReportKind::Full(
                                lsp_types::FullDocumentDiagnosticReport {
                                    result_id: Some("rust-analyzer".to_owned()),
                                    items,
                                },
                            ),
                        )
                    })
                    .collect()
            }),
        }),
    ))
}

pub(crate) fn handle_document_symbol(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentSymbolParams,
) -> anyhow::Result<Option<lsp_types::DocumentSymbolResponse>> {
    let _p = tracing::info_span!("handle_document_symbol").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;

    let mut parents: Vec<(lsp_types::DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in snap.analysis.file_structure(file_id)? {
        let mut tags = Vec::new();
        if symbol.deprecated {
            tags.push(SymbolTag::DEPRECATED)
        };

        #[allow(deprecated)]
        let doc_symbol = lsp_types::DocumentSymbol {
            name: symbol.label,
            detail: symbol.detail,
            kind: to_proto::structure_node_kind(symbol.kind),
            tags: Some(tags),
            deprecated: Some(symbol.deprecated),
            range: to_proto::range(&line_index, symbol.node_range),
            selection_range: to_proto::range(&line_index, symbol.navigation_range),
            children: None,
        };
        parents.push((doc_symbol, symbol.parent));
    }

    // Builds hierarchy from a flat list, in reverse order (so that indices
    // makes sense)
    let document_symbols = {
        let mut acc = Vec::new();
        while let Some((mut node, parent_idx)) = parents.pop() {
            if let Some(children) = &mut node.children {
                children.reverse();
            }
            let parent = match parent_idx {
                None => &mut acc,
                Some(i) => parents[i].0.children.get_or_insert_with(Vec::new),
            };
            parent.push(node);
        }
        acc.reverse();
        acc
    };

    let res = if snap.config.hierarchical_symbols() {
        document_symbols.into()
    } else {
        let url = to_proto::url(&snap, file_id);
        let mut symbol_information = Vec::<SymbolInformation>::new();
        for symbol in document_symbols {
            flatten_document_symbol(&symbol, None, &url, &mut symbol_information);
        }
        symbol_information.into()
    };
    return Ok(Some(res));

    fn flatten_document_symbol(
        symbol: &lsp_types::DocumentSymbol,
        container_name: Option<String>,
        url: &Url,
        res: &mut Vec<SymbolInformation>,
    ) {
        #[allow(deprecated)]
        res.push(SymbolInformation {
            name: symbol.name.clone(),
            kind: symbol.kind,
            tags: symbol.tags.clone(),
            deprecated: symbol.deprecated,
            location: Location::new(url.clone(), symbol.range),
            container_name,
        });

        for child in symbol.children.iter().flatten() {
            flatten_document_symbol(child, Some(symbol.name.clone()), url, res);
        }
    }
}

pub(crate) fn handle_workspace_symbol(
    snap: GlobalStateSnapshot,
    params: WorkspaceSymbolParams,
) -> anyhow::Result<Option<lsp_types::WorkspaceSymbolResponse>> {
    let _p = tracing::info_span!("handle_workspace_symbol").entered();

    let config = snap.config.workspace_symbol(None);
    let (all_symbols, libs) = decide_search_scope_and_kind(&params, &config);

    let query = {
        let query: String = params.query.chars().filter(|&c| c != '#' && c != '*').collect();
        let mut q = Query::new(query);
        if !all_symbols {
            q.only_types();
        }
        if libs {
            q.libs();
        }
        if config.search_exclude_imports {
            q.exclude_imports();
        }
        q
    };
    let mut res = exec_query(&snap, query, config.search_limit)?;
    if res.is_empty() && !all_symbols {
        res = exec_query(&snap, Query::new(params.query), config.search_limit)?;
    }

    return Ok(Some(lsp_types::WorkspaceSymbolResponse::Nested(res)));

    fn decide_search_scope_and_kind(
        params: &WorkspaceSymbolParams,
        config: &WorkspaceSymbolConfig,
    ) -> (bool, bool) {
        // Support old-style parsing of markers in the query.
        let mut all_symbols = params.query.contains('#');
        let mut libs = params.query.contains('*');

        // If no explicit marker was set, check request params. If that's also empty
        // use global config.
        if !all_symbols {
            let search_kind = match params.search_kind {
                Some(ref search_kind) => search_kind,
                None => &config.search_kind,
            };
            all_symbols = match search_kind {
                lsp_ext::WorkspaceSymbolSearchKind::OnlyTypes => false,
                lsp_ext::WorkspaceSymbolSearchKind::AllSymbols => true,
            }
        }

        if !libs {
            let search_scope = match params.search_scope {
                Some(ref search_scope) => search_scope,
                None => &config.search_scope,
            };
            libs = match search_scope {
                lsp_ext::WorkspaceSymbolSearchScope::Workspace => false,
                lsp_ext::WorkspaceSymbolSearchScope::WorkspaceAndDependencies => true,
            }
        }

        (all_symbols, libs)
    }

    fn exec_query(
        snap: &GlobalStateSnapshot,
        query: Query,
        limit: usize,
    ) -> anyhow::Result<Vec<lsp_types::WorkspaceSymbol>> {
        let mut res = Vec::new();
        for nav in snap.analysis.symbol_search(query, limit)? {
            let container_name = nav.container_name.as_ref().map(|v| v.to_string());

            let info = lsp_types::WorkspaceSymbol {
                name: match &nav.alias {
                    Some(alias) => format!("{} (alias for {})", alias, nav.name),
                    None => format!("{}", nav.name),
                },
                kind: nav
                    .kind
                    .map(to_proto::symbol_kind)
                    .unwrap_or(lsp_types::SymbolKind::VARIABLE),
                // FIXME: Set deprecation
                tags: None,
                container_name,
                location: lsp_types::OneOf::Left(to_proto::location_from_nav(snap, nav)?),
                data: None,
            };
            res.push(info);
        }
        Ok(res)
    }
}

pub(crate) fn handle_will_rename_files(
    snap: GlobalStateSnapshot,
    params: lsp_types::RenameFilesParams,
) -> anyhow::Result<Option<lsp_types::WorkspaceEdit>> {
    let _p = tracing::info_span!("handle_will_rename_files").entered();

    let source_changes: Vec<SourceChange> = params
        .files
        .into_iter()
        .filter_map(|file_rename| {
            let from = Url::parse(&file_rename.old_uri).ok()?;
            let to = Url::parse(&file_rename.new_uri).ok()?;

            let from_path = from.to_file_path().ok()?;
            let to_path = to.to_file_path().ok()?;

            // Limit to single-level moves for now.
            match (from_path.parent(), to_path.parent()) {
                (Some(p1), Some(p2)) if p1 == p2 => {
                    if from_path.is_dir() {
                        // add '/' to end of url -- from `file://path/to/folder` to `file://path/to/folder/`
                        let mut old_folder_name = from_path.file_stem()?.to_str()?.to_owned();
                        old_folder_name.push('/');
                        let from_with_trailing_slash = from.join(&old_folder_name).ok()?;

                        let imitate_from_url = from_with_trailing_slash.join("mod.rs").ok()?;
                        let new_file_name = to_path.file_name()?.to_str()?;
                        Some((
                            snap.url_to_file_id(&imitate_from_url).ok()?,
                            new_file_name.to_owned(),
                        ))
                    } else {
                        let old_name = from_path.file_stem()?.to_str()?;
                        let new_name = to_path.file_stem()?.to_str()?;
                        match (old_name, new_name) {
                            ("mod", _) => None,
                            (_, "mod") => None,
                            _ => Some((snap.url_to_file_id(&from).ok()?, new_name.to_owned())),
                        }
                    }
                }
                _ => None,
            }
        })
        .filter_map(|(file_id, new_name)| {
            snap.analysis.will_rename_file(file_id?, &new_name).ok()?
        })
        .collect();

    // Drop file system edits since we're just renaming things on the same level
    let mut source_changes = source_changes.into_iter();
    let mut source_change = source_changes.next().unwrap_or_default();
    source_change.file_system_edits.clear();
    // no collect here because we want to merge text edits on same file ids
    source_change.extend(source_changes.flat_map(|it| it.source_file_edits));
    if source_change.source_file_edits.is_empty() {
        Ok(None)
    } else {
        Ok(Some(to_proto::workspace_edit(&snap, source_change)?))
    }
}

pub(crate) fn handle_goto_definition(
    snap: GlobalStateSnapshot,
    params: lsp_types::GotoDefinitionParams,
) -> anyhow::Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = tracing::info_span!("handle_goto_definition").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);
    let nav_info = match snap.analysis.goto_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_declaration(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoDeclarationParams,
) -> anyhow::Result<Option<lsp_types::request::GotoDeclarationResponse>> {
    let _p = tracing::info_span!("handle_goto_declaration").entered();
    let position = try_default!(from_proto::file_position(
        &snap,
        params.text_document_position_params.clone()
    )?);
    let nav_info = match snap.analysis.goto_declaration(position)? {
        None => return handle_goto_definition(snap, params),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_implementation(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoImplementationParams,
) -> anyhow::Result<Option<lsp_types::request::GotoImplementationResponse>> {
    let _p = tracing::info_span!("handle_goto_implementation").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);
    let nav_info = match snap.analysis.goto_implementation(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_type_definition(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoTypeDefinitionParams,
) -> anyhow::Result<Option<lsp_types::request::GotoTypeDefinitionResponse>> {
    let _p = tracing::info_span!("handle_goto_type_definition").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);
    let nav_info = match snap.analysis.goto_type_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_parent_module(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = tracing::info_span!("handle_parent_module").entered();
    if let Ok(file_path) = &params.text_document.uri.to_file_path() {
        if file_path.file_name().unwrap_or_default() == "Cargo.toml" {
            // search workspaces for parent packages or fallback to workspace root
            let abs_path_buf = match Utf8PathBuf::from_path_buf(file_path.to_path_buf())
                .ok()
                .map(AbsPathBuf::try_from)
            {
                Some(Ok(abs_path_buf)) => abs_path_buf,
                _ => return Ok(None),
            };

            let manifest_path = match ManifestPath::try_from(abs_path_buf).ok() {
                Some(manifest_path) => manifest_path,
                None => return Ok(None),
            };

            let links: Vec<LocationLink> = snap
                .workspaces
                .iter()
                .filter_map(|ws| match &ws.kind {
                    ProjectWorkspaceKind::Cargo { cargo, .. }
                    | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, _, _)), .. } => {
                        cargo.parent_manifests(&manifest_path)
                    }
                    _ => None,
                })
                .flatten()
                .map(|parent_manifest_path| LocationLink {
                    origin_selection_range: None,
                    target_uri: to_proto::url_from_abs_path(&parent_manifest_path),
                    target_range: Range::default(),
                    target_selection_range: Range::default(),
                })
                .collect::<_>();
            return Ok(Some(links.into()));
        }

        // check if invoked at the crate root
        let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
        let crate_id = match snap.analysis.crates_for(file_id)?.first() {
            Some(&crate_id) => crate_id,
            None => return Ok(None),
        };
        let cargo_spec = match TargetSpec::for_file(&snap, file_id)? {
            Some(TargetSpec::Cargo(it)) => it,
            Some(TargetSpec::ProjectJson(_)) | None => return Ok(None),
        };

        if snap.analysis.crate_root(crate_id)? == file_id {
            let cargo_toml_url = to_proto::url_from_abs_path(&cargo_spec.cargo_toml);
            let res = vec![LocationLink {
                origin_selection_range: None,
                target_uri: cargo_toml_url,
                target_range: Range::default(),
                target_selection_range: Range::default(),
            }]
            .into();
            return Ok(Some(res));
        }
    }

    // locate parent module by semantics
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let navs = snap.analysis.parent_module(position)?;
    let res = to_proto::goto_definition_response(&snap, None, navs)?;
    Ok(Some(res))
}

pub(crate) fn handle_child_modules(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = tracing::info_span!("handle_child_modules").entered();
    // locate child module by semantics
    let position = try_default!(from_proto::file_position(&snap, params)?);
    let navs = snap.analysis.child_modules(position)?;
    let res = to_proto::goto_definition_response(&snap, None, navs)?;
    Ok(Some(res))
}

pub(crate) fn handle_runnables(
    snap: GlobalStateSnapshot,
    params: lsp_ext::RunnablesParams,
) -> anyhow::Result<Vec<lsp_ext::Runnable>> {
    let _p = tracing::info_span!("handle_runnables").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let source_root = snap.analysis.source_root_id(file_id).ok();
    let line_index = snap.file_line_index(file_id)?;
    let offset = params.position.and_then(|it| from_proto::offset(&line_index, it).ok());
    let target_spec = TargetSpec::for_file(&snap, file_id)?;

    let mut res = Vec::new();
    for runnable in snap.analysis.runnables(file_id)? {
        if should_skip_for_offset(&runnable, offset)
            || should_skip_target(&runnable, target_spec.as_ref())
        {
            continue;
        }

        let update_test = runnable.update_test;
        if let Some(mut runnable) = to_proto::runnable(&snap, runnable)? {
            if let Some(runnable) = to_proto::make_update_runnable(&runnable, update_test) {
                res.push(runnable);
            }

            if let lsp_ext::RunnableArgs::Cargo(r) = &mut runnable.args
                && let Some(TargetSpec::Cargo(CargoTargetSpec {
                    sysroot_root: Some(sysroot_root),
                    ..
                })) = &target_spec
            {
                r.environment.insert("RUSTC_TOOLCHAIN".to_owned(), sysroot_root.to_string());
            };

            res.push(runnable);
        }
    }

    // Add `cargo check` and `cargo test` for all targets of the whole package
    let config = snap.config.runnables(source_root);
    match target_spec {
        Some(TargetSpec::Cargo(spec)) => {
            let is_crate_no_std = snap.analysis.is_crate_no_std(spec.crate_id)?;
            for cmd in ["check", "run", "test"] {
                if cmd == "run" && spec.target_kind != TargetKind::Bin {
                    continue;
                }
                let cwd = if cmd != "test" || spec.target_kind == TargetKind::Bin {
                    spec.workspace_root.clone()
                } else {
                    spec.cargo_toml.parent().to_path_buf()
                };
                let mut cargo_args =
                    vec![cmd.to_owned(), "--package".to_owned(), spec.package.clone()];
                let all_targets = cmd != "run" && !is_crate_no_std;
                if all_targets {
                    cargo_args.push("--all-targets".to_owned());
                }
                cargo_args.extend(config.cargo_extra_args.iter().cloned());
                res.push(lsp_ext::Runnable {
                    label: format!(
                        "cargo {cmd} -p {}{all_targets}",
                        spec.package,
                        all_targets = if all_targets { " --all-targets" } else { "" }
                    ),
                    location: None,
                    kind: lsp_ext::RunnableKind::Cargo,
                    args: lsp_ext::RunnableArgs::Cargo(lsp_ext::CargoRunnableArgs {
                        workspace_root: Some(spec.workspace_root.clone().into()),
                        cwd: cwd.into(),
                        override_cargo: config.override_cargo.clone(),
                        cargo_args,
                        executable_args: Vec::new(),
                        environment: spec
                            .sysroot_root
                            .as_ref()
                            .map(|root| ("RUSTC_TOOLCHAIN".to_owned(), root.to_string()))
                            .into_iter()
                            .collect(),
                    }),
                })
            }
        }
        Some(TargetSpec::ProjectJson(_)) => {}
        None => {
            if !snap.config.linked_or_discovered_projects().is_empty()
                && let Some(path) = snap.file_id_to_file_path(file_id).parent()
            {
                let mut cargo_args = vec!["check".to_owned(), "--workspace".to_owned()];
                cargo_args.extend(config.cargo_extra_args.iter().cloned());
                res.push(lsp_ext::Runnable {
                    label: "cargo check --workspace".to_owned(),
                    location: None,
                    kind: lsp_ext::RunnableKind::Cargo,
                    args: lsp_ext::RunnableArgs::Cargo(lsp_ext::CargoRunnableArgs {
                        workspace_root: None,
                        cwd: path.as_path().unwrap().to_path_buf().into(),
                        override_cargo: config.override_cargo,
                        cargo_args,
                        executable_args: Vec::new(),
                        environment: Default::default(),
                    }),
                });
            };
        }
    }
    Ok(res)
}

fn should_skip_for_offset(runnable: &Runnable, offset: Option<TextSize>) -> bool {
    match offset {
        None => false,
        _ if matches!(&runnable.kind, RunnableKind::TestMod { .. }) => false,
        Some(offset) => !runnable.nav.full_range.contains_inclusive(offset),
    }
}

pub(crate) fn handle_related_tests(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Vec<lsp_ext::TestInfo>> {
    let _p = tracing::info_span!("handle_related_tests").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);

    let tests = snap.analysis.related_tests(position, None)?;
    let mut res = Vec::new();
    for it in tests {
        if let Ok(Some(runnable)) = to_proto::runnable(&snap, it) {
            res.push(lsp_ext::TestInfo { runnable })
        }
    }

    Ok(res)
}

pub(crate) fn handle_completion(
    snap: GlobalStateSnapshot,
    lsp_types::CompletionParams {
        text_document_position,
        context,
        ..
    }: lsp_types::CompletionParams,
) -> anyhow::Result<Option<lsp_types::CompletionResponse>> {
    let _p = tracing::info_span!("handle_completion").entered();
    let mut position =
        try_default!(from_proto::file_position(&snap, text_document_position.clone())?);
    let line_index = snap.file_line_index(position.file_id)?;
    let completion_trigger_character =
        context.and_then(|ctx| ctx.trigger_character).and_then(|s| s.chars().next());

    let source_root = snap.analysis.source_root_id(position.file_id)?;
    let completion_config = &snap.config.completion(Some(source_root));
    // FIXME: We should fix up the position when retrying the cancelled request instead
    position.offset = position.offset.min(line_index.index.len());
    let items = match snap.analysis.completions(
        completion_config,
        position,
        completion_trigger_character,
    )? {
        None => return Ok(None),
        Some(items) => items,
    };

    let items = to_proto::completion_items(
        &snap.config,
        &completion_config.fields_to_resolve,
        &line_index,
        snap.file_version(position.file_id),
        text_document_position,
        completion_trigger_character,
        items,
    );

    let completion_list = lsp_types::CompletionList { is_incomplete: true, items };
    Ok(Some(completion_list.into()))
}

pub(crate) fn handle_completion_resolve(
    snap: GlobalStateSnapshot,
    mut original_completion: CompletionItem,
) -> anyhow::Result<CompletionItem> {
    let _p = tracing::info_span!("handle_completion_resolve").entered();

    if !all_edits_are_disjoint(&original_completion, &[]) {
        return Err(invalid_params_error(
            "Received a completion with overlapping edits, this is not LSP-compliant".to_owned(),
        )
        .into());
    }

    let Some(data) = original_completion.data.take() else {
        return Ok(original_completion);
    };

    let resolve_data: lsp_ext::CompletionResolveData = serde_json::from_value(data)?;

    let file_id = from_proto::file_id(&snap, &resolve_data.position.text_document.uri)?
        .expect("we never provide completions for excluded files");
    let line_index = snap.file_line_index(file_id)?;
    // FIXME: We should fix up the position when retrying the cancelled request instead
    let Ok(offset) = from_proto::offset(&line_index, resolve_data.position.position) else {
        return Ok(original_completion);
    };
    let source_root = snap.analysis.source_root_id(file_id)?;

    let mut forced_resolve_completions_config = snap.config.completion(Some(source_root));
    forced_resolve_completions_config.fields_to_resolve = CompletionFieldsToResolve::empty();

    let position = FilePosition { file_id, offset };
    let Some(completions) = snap.analysis.completions(
        &forced_resolve_completions_config,
        position,
        resolve_data.trigger_character,
    )?
    else {
        return Ok(original_completion);
    };
    let Ok(resolve_data_hash) = BASE64_STANDARD.decode(resolve_data.hash) else {
        return Ok(original_completion);
    };

    let Some(corresponding_completion) = completions.into_iter().find(|completion_item| {
        // Avoid computing hashes for items that obviously do not match
        // r-a might append a detail-based suffix to the label, so we cannot check for equality
        original_completion.label.starts_with(completion_item.label.primary.as_str())
            && resolve_data_hash == completion_item_hash(completion_item, resolve_data.for_ref)
    }) else {
        return Ok(original_completion);
    };

    let mut resolved_completions = to_proto::completion_items(
        &snap.config,
        &forced_resolve_completions_config.fields_to_resolve,
        &line_index,
        snap.file_version(position.file_id),
        resolve_data.position,
        resolve_data.trigger_character,
        vec![corresponding_completion],
    );
    let Some(mut resolved_completion) = resolved_completions.pop() else {
        return Ok(original_completion);
    };

    if !resolve_data.imports.is_empty() {
        let additional_edits = snap
            .analysis
            .resolve_completion_edits(
                &forced_resolve_completions_config,
                position,
                resolve_data.imports.into_iter().map(|import| import.full_import_path),
            )?
            .into_iter()
            .flat_map(|edit| edit.into_iter().map(|indel| to_proto::text_edit(&line_index, indel)))
            .collect::<Vec<_>>();

        if !all_edits_are_disjoint(&resolved_completion, &additional_edits) {
            return Err(LspError::new(
                ErrorCode::InternalError as i32,
                "Import edit overlaps with the original completion edits, this is not LSP-compliant"
                    .into(),
            )
            .into());
        }

        if let Some(original_additional_edits) = resolved_completion.additional_text_edits.as_mut()
        {
            original_additional_edits.extend(additional_edits)
        } else {
            resolved_completion.additional_text_edits = Some(additional_edits);
        }
    }

    Ok(resolved_completion)
}

pub(crate) fn handle_folding_range(
    snap: GlobalStateSnapshot,
    params: FoldingRangeParams,
) -> anyhow::Result<Option<Vec<FoldingRange>>> {
    let _p = tracing::info_span!("handle_folding_range").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let folds = snap.analysis.folding_ranges(file_id)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;
    let line_folding_only = snap.config.line_folding_only();
    let res = folds
        .into_iter()
        .map(|it| to_proto::folding_range(&text, &line_index, line_folding_only, it))
        .collect();
    Ok(Some(res))
}

pub(crate) fn handle_signature_help(
    snap: GlobalStateSnapshot,
    params: lsp_types::SignatureHelpParams,
) -> anyhow::Result<Option<lsp_types::SignatureHelp>> {
    let _p = tracing::info_span!("handle_signature_help").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);
    let help = match snap.analysis.signature_help(position)? {
        Some(it) => it,
        None => return Ok(None),
    };
    let config = snap.config.call_info();
    let res = to_proto::signature_help(help, config, snap.config.signature_help_label_offsets());
    Ok(Some(res))
}

pub(crate) fn handle_hover(
    snap: GlobalStateSnapshot,
    params: lsp_ext::HoverParams,
) -> anyhow::Result<Option<lsp_ext::Hover>> {
    let _p = tracing::info_span!("handle_hover").entered();
    let range = match params.position {
        PositionOrRange::Position(position) => Range::new(position, position),
        PositionOrRange::Range(range) => range,
    };
    let file_range = try_default!(from_proto::file_range(&snap, &params.text_document, range)?);

    let hover = snap.config.hover();
    let info = match snap.analysis.hover(&hover, file_range)? {
        None => return Ok(None),
        Some(info) => info,
    };

    let line_index = snap.file_line_index(file_range.file_id)?;
    let range = to_proto::range(&line_index, info.range);
    let markup_kind = hover.format;
    let hover = lsp_ext::Hover {
        hover: lsp_types::Hover {
            contents: HoverContents::Markup(to_proto::markup_content(
                info.info.markup,
                markup_kind,
            )),
            range: Some(range),
        },
        actions: if snap.config.hover_actions().none() {
            Vec::new()
        } else {
            prepare_hover_actions(&snap, &info.info.actions)
        },
    };

    Ok(Some(hover))
}

pub(crate) fn handle_prepare_rename(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<PrepareRenameResponse>> {
    let _p = tracing::info_span!("handle_prepare_rename").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);

    let change = snap.analysis.prepare_rename(position)?.map_err(to_proto::rename_error)?;

    let line_index = snap.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, change.range);
    Ok(Some(PrepareRenameResponse::Range(range)))
}

pub(crate) fn handle_rename(
    snap: GlobalStateSnapshot,
    params: RenameParams,
) -> anyhow::Result<Option<WorkspaceEdit>> {
    let _p = tracing::info_span!("handle_rename").entered();
    let position = try_default!(from_proto::file_position(&snap, params.text_document_position)?);

    let mut change =
        snap.analysis.rename(position, &params.new_name)?.map_err(to_proto::rename_error)?;

    // this is kind of a hack to prevent double edits from happening when moving files
    // When a module gets renamed by renaming the mod declaration this causes the file to move
    // which in turn will trigger a WillRenameFiles request to the server for which we reply with a
    // a second identical set of renames, the client will then apply both edits causing incorrect edits
    // with this we only emit source_file_edits in the WillRenameFiles response which will do the rename instead
    // See https://github.com/microsoft/vscode-languageserver-node/issues/752 for more info
    if !change.file_system_edits.is_empty() && snap.config.will_rename() {
        change.source_file_edits.clear();
    }

    let workspace_edit = to_proto::workspace_edit(&snap, change)?;

    if let Some(lsp_types::DocumentChanges::Operations(ops)) =
        workspace_edit.document_changes.as_ref()
    {
        for op in ops {
            if let lsp_types::DocumentChangeOperation::Op(doc_change_op) = op {
                resource_ops_supported(&snap.config, resolve_resource_op(doc_change_op))?
            }
        }
    }

    Ok(Some(workspace_edit))
}

pub(crate) fn handle_references(
    snap: GlobalStateSnapshot,
    params: lsp_types::ReferenceParams,
) -> anyhow::Result<Option<Vec<Location>>> {
    let _p = tracing::info_span!("handle_references").entered();
    let position = try_default!(from_proto::file_position(&snap, params.text_document_position)?);

    let exclude_imports = snap.config.find_all_refs_exclude_imports();
    let exclude_tests = snap.config.find_all_refs_exclude_tests();

    let Some(refs) = snap.analysis.find_all_refs(position, None)? else {
        return Ok(None);
    };

    let include_declaration = params.context.include_declaration;
    let locations = refs
        .into_iter()
        .flat_map(|refs| {
            let decl = if include_declaration {
                refs.declaration.map(|decl| FileRange {
                    file_id: decl.nav.file_id,
                    range: decl.nav.focus_or_full_range(),
                })
            } else {
                None
            };
            refs.references
                .into_iter()
                .flat_map(|(file_id, refs)| {
                    refs.into_iter()
                        .filter(|&(_, category)| {
                            (!exclude_imports || !category.contains(ReferenceCategory::IMPORT))
                                && (!exclude_tests || !category.contains(ReferenceCategory::TEST))
                        })
                        .map(move |(range, _)| FileRange { file_id, range })
                })
                .chain(decl)
        })
        .unique()
        .filter_map(|frange| to_proto::location(&snap, frange).ok())
        .collect();

    Ok(Some(locations))
}

pub(crate) fn handle_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentFormattingParams,
) -> anyhow::Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = tracing::info_span!("handle_formatting").entered();

    run_rustfmt(&snap, params.text_document, None)
}

pub(crate) fn handle_range_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentRangeFormattingParams,
) -> anyhow::Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = tracing::info_span!("handle_range_formatting").entered();

    run_rustfmt(&snap, params.text_document, Some(params.range))
}

pub(crate) fn handle_code_action(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeActionParams,
) -> anyhow::Result<Option<Vec<lsp_ext::CodeAction>>> {
    let _p = tracing::info_span!("handle_code_action").entered();

    if !snap.config.code_action_literals() {
        // We intentionally don't support command-based actions, as those either
        // require either custom client-code or server-initiated edits. Server
        // initiated edits break causality, so we avoid those.
        return Ok(None);
    }

    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    let frange = try_default!(from_proto::file_range(&snap, &params.text_document, params.range)?);
    let source_root = snap.analysis.source_root_id(file_id)?;

    let mut assists_config = snap.config.assist(Some(source_root));
    assists_config.allowed = params
        .context
        .only
        .clone()
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let mut res: Vec<lsp_ext::CodeAction> = Vec::new();

    let code_action_resolve_cap = snap.config.code_action_resolve();
    let resolve = if code_action_resolve_cap {
        AssistResolveStrategy::None
    } else {
        AssistResolveStrategy::All
    };
    let assists = snap.analysis.assists_with_fixes(
        &assists_config,
        &snap.config.diagnostic_fixes(Some(source_root)),
        resolve,
        frange,
    )?;
    for (index, assist) in assists.into_iter().enumerate() {
        let resolve_data = if code_action_resolve_cap {
            Some((index, params.clone(), snap.file_version(file_id)))
        } else {
            None
        };
        let code_action = to_proto::code_action(&snap, assist, resolve_data)?;

        // Check if the client supports the necessary `ResourceOperation`s.
        let changes = code_action.edit.as_ref().and_then(|it| it.document_changes.as_ref());
        if let Some(changes) = changes {
            for change in changes {
                if let lsp_ext::SnippetDocumentChangeOperation::Op(res_op) = change {
                    resource_ops_supported(&snap.config, resolve_resource_op(res_op))?
                }
            }
        }

        res.push(code_action)
    }

    // Fixes from `cargo check`.
    for fix in snap
        .check_fixes
        .iter()
        .flat_map(|it| it.values())
        .filter_map(|it| it.get(&frange.file_id))
        .flatten()
    {
        // FIXME: this mapping is awkward and shouldn't exist. Refactor
        // `snap.check_fixes` to not convert to LSP prematurely.
        let intersect_fix_range = fix
            .ranges
            .iter()
            .copied()
            .filter_map(|range| from_proto::text_range(&line_index, range).ok())
            .any(|fix_range| fix_range.intersect(frange.range).is_some());
        if intersect_fix_range {
            res.push(fix.action.clone());
        }
    }

    Ok(Some(res))
}

pub(crate) fn handle_code_action_resolve(
    snap: GlobalStateSnapshot,
    mut code_action: lsp_ext::CodeAction,
) -> anyhow::Result<lsp_ext::CodeAction> {
    let _p = tracing::info_span!("handle_code_action_resolve").entered();
    let Some(params) = code_action.data.take() else {
        return Ok(code_action);
    };

    let file_id = from_proto::file_id(&snap, &params.code_action_params.text_document.uri)?
        .expect("we never provide code actions for excluded files");
    if snap.file_version(file_id) != params.version {
        return Err(invalid_params_error("stale code action".to_owned()).into());
    }
    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.code_action_params.range)?;
    let frange = FileRange { file_id, range };
    let source_root = snap.analysis.source_root_id(file_id)?;

    let mut assists_config = snap.config.assist(Some(source_root));
    assists_config.allowed = params
        .code_action_params
        .context
        .only
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let (assist_index, assist_resolve) = match parse_action_id(&params.id) {
        Ok(parsed_data) => parsed_data,
        Err(e) => {
            return Err(invalid_params_error(format!(
                "Failed to parse action id string '{}': {e}",
                params.id
            ))
            .into());
        }
    };

    let expected_assist_id = assist_resolve.assist_id.clone();
    let expected_kind = assist_resolve.assist_kind;

    let assists = snap.analysis.assists_with_fixes(
        &assists_config,
        &snap.config.diagnostic_fixes(Some(source_root)),
        AssistResolveStrategy::Single(assist_resolve),
        frange,
    )?;

    let assist = match assists.get(assist_index) {
        Some(assist) => assist,
        None => return Err(invalid_params_error(format!(
            "Failed to find the assist for index {} provided by the resolve request. Resolve request assist id: {}",
            assist_index, params.id,
        ))
        .into())
    };
    if assist.id.0 != expected_assist_id || assist.id.1 != expected_kind {
        return Err(invalid_params_error(format!(
            "Mismatching assist at index {} for the resolve parameters given. Resolve request assist id: {}, actual id: {:?}.",
            assist_index, params.id, assist.id
        ))
        .into());
    }
    let ca = to_proto::code_action(&snap, assist.clone(), None)?;
    code_action.edit = ca.edit;
    code_action.command = ca.command;

    if let Some(edit) = code_action.edit.as_ref()
        && let Some(changes) = edit.document_changes.as_ref()
    {
        for change in changes {
            if let lsp_ext::SnippetDocumentChangeOperation::Op(res_op) = change {
                resource_ops_supported(&snap.config, resolve_resource_op(res_op))?
            }
        }
    }

    Ok(code_action)
}

fn parse_action_id(action_id: &str) -> anyhow::Result<(usize, SingleResolve), String> {
    let id_parts = action_id.split(':').collect::<Vec<_>>();
    match id_parts.as_slice() {
        [assist_id_string, assist_kind_string, index_string, subtype_str] => {
            let assist_kind: AssistKind = assist_kind_string.parse()?;
            let index: usize = match index_string.parse() {
                Ok(index) => index,
                Err(e) => return Err(format!("Incorrect index string: {e}")),
            };
            let assist_subtype = subtype_str.parse::<usize>().ok();
            Ok((
                index,
                SingleResolve {
                    assist_id: assist_id_string.to_string(),
                    assist_kind,
                    assist_subtype,
                },
            ))
        }
        _ => Err("Action id contains incorrect number of segments".to_owned()),
    }
}

pub(crate) fn handle_code_lens(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeLensParams,
) -> anyhow::Result<Option<Vec<CodeLens>>> {
    let _p = tracing::info_span!("handle_code_lens").entered();

    let lens_config = snap.config.lens();
    if lens_config.none() {
        // early return before any db query!
        return Ok(Some(Vec::default()));
    }

    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let target_spec = TargetSpec::for_file(&snap, file_id)?;

    let annotations = snap.analysis.annotations(
        &AnnotationConfig {
            binary_target: target_spec
                .map(|spec| {
                    matches!(
                        spec.target_kind(),
                        TargetKind::Bin | TargetKind::Example | TargetKind::Test
                    )
                })
                .unwrap_or(false),
            annotate_runnables: lens_config.runnable(),
            annotate_impls: lens_config.implementations,
            annotate_references: lens_config.refs_adt,
            annotate_method_references: lens_config.method_refs,
            annotate_enum_variant_references: lens_config.enum_variant_refs,
            location: lens_config.location.into(),
        },
        file_id,
    )?;

    let mut res = Vec::new();
    for a in annotations {
        to_proto::code_lens(&mut res, &snap, a)?;
    }

    Ok(Some(res))
}

pub(crate) fn handle_code_lens_resolve(
    snap: GlobalStateSnapshot,
    mut code_lens: CodeLens,
) -> anyhow::Result<CodeLens> {
    let Some(data) = code_lens.data.take() else {
        return Ok(code_lens);
    };
    let resolve = serde_json::from_value::<lsp_ext::CodeLensResolveData>(data)?;
    let Some(annotation) = from_proto::annotation(&snap, code_lens.range, resolve)? else {
        return Ok(code_lens);
    };
    let annotation = snap.analysis.resolve_annotation(annotation)?;

    let mut acc = Vec::new();
    to_proto::code_lens(&mut acc, &snap, annotation)?;

    let mut res = match acc.pop() {
        Some(it) if acc.is_empty() => it,
        _ => {
            never!();
            code_lens
        }
    };
    res.data = None;

    Ok(res)
}

pub(crate) fn handle_document_highlight(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentHighlightParams,
) -> anyhow::Result<Option<Vec<lsp_types::DocumentHighlight>>> {
    let _p = tracing::info_span!("handle_document_highlight").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);
    let line_index = snap.file_line_index(position.file_id)?;
    let source_root = snap.analysis.source_root_id(position.file_id)?;

    let refs = match snap
        .analysis
        .highlight_related(snap.config.highlight_related(Some(source_root)), position)?
    {
        None => return Ok(None),
        Some(refs) => refs,
    };
    let res = refs
        .into_iter()
        .map(|ide::HighlightedRange { range, category }| lsp_types::DocumentHighlight {
            range: to_proto::range(&line_index, range),
            kind: to_proto::document_highlight_kind(category),
        })
        .collect();
    Ok(Some(res))
}

pub(crate) fn handle_ssr(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SsrParams,
) -> anyhow::Result<lsp_types::WorkspaceEdit> {
    let _p = tracing::info_span!("handle_ssr").entered();
    let selections = try_default!(
        params
            .selections
            .iter()
            .map(|range| from_proto::file_range(&snap, &params.position.text_document, *range))
            .collect::<Result<Option<Vec<_>>, _>>()?
    );
    let position = try_default!(from_proto::file_position(&snap, params.position)?);
    let source_change = snap.analysis.structural_search_replace(
        &params.query,
        params.parse_only,
        position,
        selections,
    )??;
    to_proto::workspace_edit(&snap, source_change).map_err(Into::into)
}

pub(crate) fn handle_inlay_hints(
    snap: GlobalStateSnapshot,
    params: InlayHintParams,
) -> anyhow::Result<Option<Vec<InlayHint>>> {
    let _p = tracing::info_span!("handle_inlay_hints").entered();
    let document_uri = &params.text_document.uri;
    let FileRange { file_id, range } = try_default!(from_proto::file_range(
        &snap,
        &TextDocumentIdentifier::new(document_uri.to_owned()),
        params.range,
    )?);
    let line_index = snap.file_line_index(file_id)?;
    let range = TextRange::new(
        range.start().min(line_index.index.len()),
        range.end().min(line_index.index.len()),
    );

    let inlay_hints_config = snap.config.inlay_hints();
    Ok(Some(
        snap.analysis
            .inlay_hints(&inlay_hints_config, file_id, Some(range))?
            .into_iter()
            .map(|it| {
                to_proto::inlay_hint(
                    &snap,
                    &inlay_hints_config.fields_to_resolve,
                    &line_index,
                    file_id,
                    it,
                )
            })
            .collect::<Cancellable<Vec<_>>>()?,
    ))
}

pub(crate) fn handle_inlay_hints_resolve(
    snap: GlobalStateSnapshot,
    mut original_hint: InlayHint,
) -> anyhow::Result<InlayHint> {
    let _p = tracing::info_span!("handle_inlay_hints_resolve").entered();

    let Some(data) = original_hint.data.take() else {
        return Ok(original_hint);
    };
    let resolve_data: lsp_ext::InlayHintResolveData = serde_json::from_value(data)?;
    let file_id = FileId::from_raw(resolve_data.file_id);
    if resolve_data.version != snap.file_version(file_id) {
        tracing::warn!("Inlay hint resolve data is outdated");
        return Ok(original_hint);
    }
    let Some(hash) = resolve_data.hash.parse().ok() else {
        return Ok(original_hint);
    };
    anyhow::ensure!(snap.file_exists(file_id), "Invalid LSP resolve data");

    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, resolve_data.resolve_range)?;

    let mut forced_resolve_inlay_hints_config = snap.config.inlay_hints();
    forced_resolve_inlay_hints_config.fields_to_resolve = InlayFieldsToResolve::empty();
    let resolve_hints = snap.analysis.inlay_hints_resolve(
        &forced_resolve_inlay_hints_config,
        file_id,
        range,
        hash,
        |hint| {
            std::hash::BuildHasher::hash_one(
                &std::hash::BuildHasherDefault::<ide_db::FxHasher>::default(),
                hint,
            )
        },
    )?;

    Ok(resolve_hints
        .and_then(|it| {
            to_proto::inlay_hint(
                &snap,
                &forced_resolve_inlay_hints_config.fields_to_resolve,
                &line_index,
                file_id,
                it,
            )
            .ok()
        })
        .filter(|hint| hint.position == original_hint.position)
        .filter(|hint| hint.kind == original_hint.kind)
        .unwrap_or(original_hint))
}

pub(crate) fn handle_call_hierarchy_prepare(
    snap: GlobalStateSnapshot,
    params: CallHierarchyPrepareParams,
) -> anyhow::Result<Option<Vec<CallHierarchyItem>>> {
    let _p = tracing::info_span!("handle_call_hierarchy_prepare").entered();
    let position =
        try_default!(from_proto::file_position(&snap, params.text_document_position_params)?);

    let nav_info = match snap.analysis.call_hierarchy(position)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let RangeInfo { range: _, info: navs } = nav_info;
    let res = navs
        .into_iter()
        .filter(|it| matches!(it.kind, Some(SymbolKind::Function | SymbolKind::Method)))
        .map(|it| to_proto::call_hierarchy_item(&snap, it))
        .collect::<Cancellable<Vec<_>>>()?;

    Ok(Some(res))
}

pub(crate) fn handle_call_hierarchy_incoming(
    snap: GlobalStateSnapshot,
    params: CallHierarchyIncomingCallsParams,
) -> anyhow::Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let _p = tracing::info_span!("handle_call_hierarchy_incoming").entered();
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = try_default!(from_proto::file_range(&snap, &doc, item.selection_range)?);
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let config = snap.config.call_hierarchy();
    let call_items = match snap.analysis.incoming_calls(config, fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id;
        let line_index = snap.file_line_index(file_id)?;
        let item = to_proto::call_hierarchy_item(&snap, call_item.target)?;
        res.push(CallHierarchyIncomingCall {
            from: item,
            from_ranges: call_item
                .ranges
                .into_iter()
                // This is the range relative to the item
                .filter(|it| it.file_id == file_id)
                .map(|it| to_proto::range(&line_index, it.range))
                .collect(),
        });
    }

    Ok(Some(res))
}

pub(crate) fn handle_call_hierarchy_outgoing(
    snap: GlobalStateSnapshot,
    params: CallHierarchyOutgoingCallsParams,
) -> anyhow::Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    let _p = tracing::info_span!("handle_call_hierarchy_outgoing").entered();
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = try_default!(from_proto::file_range(&snap, &doc, item.selection_range)?);
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };
    let line_index = snap.file_line_index(fpos.file_id)?;

    let config = snap.config.call_hierarchy();
    let call_items = match snap.analysis.outgoing_calls(config, fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let item = to_proto::call_hierarchy_item(&snap, call_item.target)?;
        res.push(CallHierarchyOutgoingCall {
            to: item,
            from_ranges: call_item
                .ranges
                .into_iter()
                // This is the range relative to the caller
                .filter(|it| it.file_id == fpos.file_id)
                .map(|it| to_proto::range(&line_index, it.range))
                .collect(),
        });
    }

    Ok(Some(res))
}

pub(crate) fn handle_semantic_tokens_full(
    snap: GlobalStateSnapshot,
    params: SemanticTokensParams,
) -> anyhow::Result<Option<SemanticTokensResult>> {
    let _p = tracing::info_span!("handle_semantic_tokens_full").entered();

    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;

    let mut highlight_config = snap.config.highlighting_config();
    // Avoid flashing a bunch of unresolved references when the proc-macro servers haven't been spawned yet.
    highlight_config.syntactic_name_ref_highlighting =
        snap.workspaces.is_empty() || !snap.proc_macros_loaded;

    let highlights = snap.analysis.highlight(highlight_config, file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(
        &text,
        &line_index,
        highlights,
        snap.config.semantics_tokens_augments_syntax_tokens(),
        snap.config.highlighting_non_standard_tokens(),
    );

    // Unconditionally cache the tokens
    snap.semantic_tokens_cache.lock().insert(params.text_document.uri, semantic_tokens.clone());

    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_semantic_tokens_full_delta(
    snap: GlobalStateSnapshot,
    params: SemanticTokensDeltaParams,
) -> anyhow::Result<Option<SemanticTokensFullDeltaResult>> {
    let _p = tracing::info_span!("handle_semantic_tokens_full_delta").entered();

    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;

    let mut highlight_config = snap.config.highlighting_config();
    // Avoid flashing a bunch of unresolved references when the proc-macro servers haven't been spawned yet.
    highlight_config.syntactic_name_ref_highlighting =
        snap.workspaces.is_empty() || !snap.proc_macros_loaded;

    let highlights = snap.analysis.highlight(highlight_config, file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(
        &text,
        &line_index,
        highlights,
        snap.config.semantics_tokens_augments_syntax_tokens(),
        snap.config.highlighting_non_standard_tokens(),
    );

    let cached_tokens = snap.semantic_tokens_cache.lock().remove(&params.text_document.uri);

    if let Some(cached_tokens @ lsp_types::SemanticTokens { result_id: Some(prev_id), .. }) =
        &cached_tokens
        && *prev_id == params.previous_result_id
    {
        let delta = to_proto::semantic_token_delta(cached_tokens, &semantic_tokens);
        snap.semantic_tokens_cache.lock().insert(params.text_document.uri, semantic_tokens);
        return Ok(Some(delta.into()));
    }

    // Clone first to keep the lock short
    let semantic_tokens_clone = semantic_tokens.clone();
    snap.semantic_tokens_cache.lock().insert(params.text_document.uri, semantic_tokens_clone);

    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_semantic_tokens_range(
    snap: GlobalStateSnapshot,
    params: SemanticTokensRangeParams,
) -> anyhow::Result<Option<SemanticTokensRangeResult>> {
    let _p = tracing::info_span!("handle_semantic_tokens_range").entered();

    let frange = try_default!(from_proto::file_range(&snap, &params.text_document, params.range)?);
    let text = snap.analysis.file_text(frange.file_id)?;
    let line_index = snap.file_line_index(frange.file_id)?;

    let mut highlight_config = snap.config.highlighting_config();
    // Avoid flashing a bunch of unresolved references when the proc-macro servers haven't been spawned yet.
    highlight_config.syntactic_name_ref_highlighting =
        snap.workspaces.is_empty() || !snap.proc_macros_loaded;

    let highlights = snap.analysis.highlight_range(highlight_config, frange)?;
    let semantic_tokens = to_proto::semantic_tokens(
        &text,
        &line_index,
        highlights,
        snap.config.semantics_tokens_augments_syntax_tokens(),
        snap.config.highlighting_non_standard_tokens(),
    );
    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_open_docs(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<ExternalDocsResponse> {
    let _p = tracing::info_span!("handle_open_docs").entered();
    let position = try_default!(from_proto::file_position(&snap, params)?);

    let ws_and_sysroot = snap.workspaces.iter().find_map(|ws| match &ws.kind {
        ProjectWorkspaceKind::Cargo { cargo, .. }
        | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, _, _)), .. } => {
            Some((cargo, &ws.sysroot))
        }
        ProjectWorkspaceKind::Json { .. } => None,
        ProjectWorkspaceKind::DetachedFile { .. } => None,
    });

    let (cargo, sysroot) = match ws_and_sysroot {
        Some((ws, sysroot)) => (Some(ws), Some(sysroot)),
        _ => (None, None),
    };

    let sysroot = sysroot.and_then(|p| p.root()).map(|it| it.as_str());
    let target_dir = cargo.map(|cargo| cargo.target_directory()).map(|p| p.as_str());

    let Ok(remote_urls) = snap.analysis.external_docs(position, target_dir, sysroot) else {
        return if snap.config.local_docs() {
            Ok(ExternalDocsResponse::WithLocal(Default::default()))
        } else {
            Ok(ExternalDocsResponse::Simple(None))
        };
    };

    let web = remote_urls.web_url.and_then(|it| Url::parse(&it).ok());
    let local = remote_urls.local_url.and_then(|it| Url::parse(&it).ok());

    if snap.config.local_docs() {
        Ok(ExternalDocsResponse::WithLocal(ExternalDocsPair { web, local }))
    } else {
        Ok(ExternalDocsResponse::Simple(web))
    }
}

pub(crate) fn handle_open_cargo_toml(
    snap: GlobalStateSnapshot,
    params: lsp_ext::OpenCargoTomlParams,
) -> anyhow::Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = tracing::info_span!("handle_open_cargo_toml").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);

    let cargo_spec = match TargetSpec::for_file(&snap, file_id)? {
        Some(TargetSpec::Cargo(it)) => it,
        Some(TargetSpec::ProjectJson(_)) | None => return Ok(None),
    };

    let cargo_toml_url = to_proto::url_from_abs_path(&cargo_spec.cargo_toml);
    let res: lsp_types::GotoDefinitionResponse =
        Location::new(cargo_toml_url, Range::default()).into();
    Ok(Some(res))
}

pub(crate) fn handle_move_item(
    snap: GlobalStateSnapshot,
    params: lsp_ext::MoveItemParams,
) -> anyhow::Result<Vec<lsp_ext::SnippetTextEdit>> {
    let _p = tracing::info_span!("handle_move_item").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let range = try_default!(from_proto::file_range(&snap, &params.text_document, params.range)?);

    let direction = match params.direction {
        lsp_ext::MoveItemDirection::Up => ide::Direction::Up,
        lsp_ext::MoveItemDirection::Down => ide::Direction::Down,
    };

    match snap.analysis.move_item(range, direction)? {
        Some(text_edit) => {
            let line_index = snap.file_line_index(file_id)?;
            Ok(to_proto::snippet_text_edit_vec(
                &line_index,
                true,
                text_edit,
                snap.config.change_annotation_support(),
            ))
        }
        None => Ok(vec![]),
    }
}

pub(crate) fn handle_view_recursive_memory_layout(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<lsp_ext::RecursiveMemoryLayout>> {
    let _p = tracing::info_span!("handle_view_recursive_memory_layout").entered();
    let file_id = try_default!(from_proto::file_id(&snap, &params.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, params.position)?;

    let res = snap.analysis.get_recursive_memory_layout(FilePosition { file_id, offset })?;
    Ok(res.map(|it| lsp_ext::RecursiveMemoryLayout {
        nodes: it
            .nodes
            .iter()
            .map(|n| lsp_ext::MemoryLayoutNode {
                item_name: n.item_name.clone(),
                typename: n.typename.clone(),
                size: n.size,
                offset: n.offset,
                alignment: n.alignment,
                parent_idx: n.parent_idx,
                children_start: n.children_start,
                children_len: n.children_len,
            })
            .collect(),
    }))
}

fn to_command_link(command: lsp_types::Command, tooltip: String) -> lsp_ext::CommandLink {
    lsp_ext::CommandLink { tooltip: Some(tooltip), command }
}

fn show_impl_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover_actions().implementations
        && snap.config.client_commands().show_reference
        && let Some(nav_data) = snap.analysis.goto_implementation(*position).unwrap_or(None)
    {
        let uri = to_proto::url(snap, position.file_id);
        let line_index = snap.file_line_index(position.file_id).ok()?;
        let position = to_proto::position(&line_index, position.offset);
        let locations: Vec<_> = nav_data
            .info
            .into_iter()
            .filter_map(|nav| to_proto::location_from_nav(snap, nav).ok())
            .collect();
        let title = to_proto::implementation_title(locations.len());
        let command = to_proto::command::show_references(title, &uri, position, locations);

        return Some(lsp_ext::CommandLinkGroup {
            commands: vec![to_command_link(command, "Go to implementations".into())],
            ..Default::default()
        });
    }
    None
}

fn show_ref_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover_actions().references
        && snap.config.client_commands().show_reference
        && let Some(ref_search_res) = snap.analysis.find_all_refs(*position, None).unwrap_or(None)
    {
        let uri = to_proto::url(snap, position.file_id);
        let line_index = snap.file_line_index(position.file_id).ok()?;
        let position = to_proto::position(&line_index, position.offset);
        let locations: Vec<_> = ref_search_res
            .into_iter()
            .flat_map(|res| res.references)
            .flat_map(|(file_id, ranges)| {
                ranges.into_iter().map(move |(range, _)| FileRange { file_id, range })
            })
            .unique()
            .filter_map(|range| to_proto::location(snap, range).ok())
            .collect();
        let title = to_proto::reference_title(locations.len());
        let command = to_proto::command::show_references(title, &uri, position, locations);

        return Some(lsp_ext::CommandLinkGroup {
            commands: vec![to_command_link(command, "Go to references".into())],
            ..Default::default()
        });
    }
    None
}

fn runnable_action_links(
    snap: &GlobalStateSnapshot,
    runnable: Runnable,
) -> Option<lsp_ext::CommandLinkGroup> {
    let hover_actions_config = snap.config.hover_actions();
    if !hover_actions_config.runnable() {
        return None;
    }

    let target_spec = TargetSpec::for_file(snap, runnable.nav.file_id).ok()?;
    if should_skip_target(&runnable, target_spec.as_ref()) {
        return None;
    }

    let client_commands_config = snap.config.client_commands();
    if !(client_commands_config.run_single || client_commands_config.debug_single) {
        return None;
    }

    let title = runnable.title();
    let update_test = runnable.update_test;
    let r = to_proto::runnable(snap, runnable).ok()??;

    let mut group = lsp_ext::CommandLinkGroup::default();

    if hover_actions_config.run && client_commands_config.run_single {
        let run_command = to_proto::command::run_single(&r, &title);
        group.commands.push(to_command_link(run_command, r.label.clone()));
    }

    if hover_actions_config.debug && client_commands_config.debug_single {
        let dbg_command = to_proto::command::debug_single(&r);
        group.commands.push(to_command_link(dbg_command, r.label.clone()));
    }

    if hover_actions_config.update_test && client_commands_config.run_single {
        let label = update_test.label();
        if let Some(r) = to_proto::make_update_runnable(&r, update_test) {
            let update_command = to_proto::command::run_single(&r, label.unwrap().as_str());
            group.commands.push(to_command_link(update_command, r.label));
        }
    }

    Some(group)
}

fn goto_type_action_links(
    snap: &GlobalStateSnapshot,
    nav_targets: &[HoverGotoTypeData],
) -> Option<lsp_ext::CommandLinkGroup> {
    if !snap.config.hover_actions().goto_type_def
        || nav_targets.is_empty()
        || !snap.config.client_commands().goto_location
    {
        return None;
    }

    Some(lsp_ext::CommandLinkGroup {
        title: Some("Go to ".into()),
        commands: nav_targets
            .iter()
            .filter_map(|it| {
                to_proto::command::goto_location(snap, &it.nav)
                    .map(|cmd| to_command_link(cmd, it.mod_path.clone()))
            })
            .collect(),
    })
}

fn prepare_hover_actions(
    snap: &GlobalStateSnapshot,
    actions: &[HoverAction],
) -> Vec<lsp_ext::CommandLinkGroup> {
    actions
        .iter()
        .filter_map(|it| match it {
            HoverAction::Implementation(position) => show_impl_command_link(snap, position),
            HoverAction::Reference(position) => show_ref_command_link(snap, position),
            HoverAction::Runnable(r) => runnable_action_links(snap, r.clone()),
            HoverAction::GoToType(targets) => goto_type_action_links(snap, targets),
        })
        .collect()
}

fn should_skip_target(runnable: &Runnable, cargo_spec: Option<&TargetSpec>) -> bool {
    match runnable.kind {
        RunnableKind::Bin => {
            // Do not suggest binary run on other target than binary
            match &cargo_spec {
                Some(spec) => !matches!(
                    spec.target_kind(),
                    TargetKind::Bin | TargetKind::Example | TargetKind::Test
                ),
                None => true,
            }
        }
        _ => false,
    }
}

fn run_rustfmt(
    snap: &GlobalStateSnapshot,
    text_document: TextDocumentIdentifier,
    range: Option<lsp_types::Range>,
) -> anyhow::Result<Option<Vec<lsp_types::TextEdit>>> {
    let file_id = try_default!(from_proto::file_id(snap, &text_document.uri)?);
    let file = snap.analysis.file_text(file_id)?;

    // Determine the edition of the crate the file belongs to (if there's multiple, we pick the
    // highest edition).
    let Ok(editions) = snap
        .analysis
        .relevant_crates_for(file_id)?
        .into_iter()
        .map(|crate_id| snap.analysis.crate_edition(crate_id))
        .collect::<Result<Vec<_>, _>>()
    else {
        return Ok(None);
    };
    let edition = editions.iter().copied().max();

    let line_index = snap.file_line_index(file_id)?;
    let source_root_id = snap.analysis.source_root_id(file_id).ok();

    // try to chdir to the file so we can respect `rustfmt.toml`
    // FIXME: use `rustfmt --config-path` once
    // https://github.com/rust-lang/rustfmt/issues/4660 gets fixed
    let current_dir = match text_document.uri.to_file_path() {
        Ok(mut path) => {
            // pop off file name
            if path.pop() && path.is_dir() { path } else { std::env::current_dir()? }
        }
        Err(_) => {
            tracing::error!(
                text_document = ?text_document.uri,
                "Unable to get path, rustfmt.toml might be ignored"
            );
            std::env::current_dir()?
        }
    };

    let mut command = match snap.config.rustfmt(source_root_id) {
        RustfmtConfig::Rustfmt { extra_args, enable_range_formatting } => {
            // FIXME: Set RUSTUP_TOOLCHAIN
            let mut cmd = toolchain::command(
                toolchain::Tool::Rustfmt.path(),
                current_dir,
                snap.config.extra_env(source_root_id),
            );
            cmd.args(extra_args);

            if let Some(edition) = edition {
                cmd.arg("--edition");
                cmd.arg(edition.to_string());
            }

            if let Some(range) = range {
                if !enable_range_formatting {
                    return Err(LspError::new(
                        ErrorCode::InvalidRequest as i32,
                        String::from(
                            "rustfmt range formatting is unstable. \
                            Opt-in by using a nightly build of rustfmt and setting \
                            `rustfmt.rangeFormatting.enable` to true in your LSP configuration",
                        ),
                    )
                    .into());
                }

                let frange = try_default!(from_proto::file_range(snap, &text_document, range)?);
                let start_line = line_index.index.line_col(frange.range.start()).line;
                let end_line = line_index.index.line_col(frange.range.end()).line;

                cmd.arg("--unstable-features");
                cmd.arg("--file-lines");
                cmd.arg(
                    json!([{
                        "file": "stdin",
                        // LineCol is 0-based, but rustfmt is 1-based.
                        "range": [start_line + 1, end_line + 1]
                    }])
                    .to_string(),
                );
            }

            cmd
        }
        RustfmtConfig::CustomCommand { command, args } => {
            let cmd = Utf8PathBuf::from(&command);
            let target_spec = TargetSpec::for_file(snap, file_id)?;
            let extra_env = snap.config.extra_env(source_root_id);
            let mut cmd = match target_spec {
                Some(TargetSpec::Cargo(_)) => {
                    // approach: if the command name contains a path separator, join it with the project root.
                    // however, if the path is absolute, joining will result in the absolute path being preserved.
                    // as a fallback, rely on $PATH-based discovery.
                    let cmd_path = if command.contains(std::path::MAIN_SEPARATOR)
                        || (cfg!(windows) && command.contains('/'))
                    {
                        snap.config.root_path().join(cmd).into()
                    } else {
                        cmd
                    };
                    toolchain::command(cmd_path, current_dir, extra_env)
                }
                _ => toolchain::command(cmd, current_dir, extra_env),
            };

            cmd.args(args);
            cmd
        }
    };

    let output = {
        let _p = tracing::info_span!("rustfmt", ?command).entered();

        let mut rustfmt = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(format!("Failed to spawn {command:?}"))?;

        rustfmt.stdin.as_mut().unwrap().write_all(file.as_bytes())?;

        rustfmt.wait_with_output()?
    };

    let captured_stdout = String::from_utf8(output.stdout)?;
    let captured_stderr = String::from_utf8(output.stderr).unwrap_or_default();

    if !output.status.success() {
        let rustfmt_not_installed =
            captured_stderr.contains("not installed") || captured_stderr.contains("not available");

        return match output.status.code() {
            Some(1) if !rustfmt_not_installed => {
                // While `rustfmt` doesn't have a specific exit code for parse errors this is the
                // likely cause exiting with 1. Most Language Servers swallow parse errors on
                // formatting because otherwise an error is surfaced to the user on top of the
                // syntax error diagnostics they're already receiving. This is especially jarring
                // if they have format on save enabled.
                tracing::warn!(
                    ?command,
                    %captured_stderr,
                    "rustfmt exited with status 1"
                );
                Ok(None)
            }
            // rustfmt panicked at lexing/parsing the file
            Some(101)
                if !rustfmt_not_installed
                    && (captured_stderr.starts_with("error[")
                        || captured_stderr.starts_with("error:")) =>
            {
                Ok(None)
            }
            _ => {
                // Something else happened - e.g. `rustfmt` is missing or caught a signal
                tracing::error!(
                    ?command,
                    %output.status,
                    %captured_stdout,
                    %captured_stderr,
                    "rustfmt failed"
                );
                Ok(None)
            }
        };
    }

    let (new_text, new_line_endings) = LineEndings::normalize(captured_stdout);

    if line_index.endings != new_line_endings {
        // If line endings are different, send the entire file.
        // Diffing would not work here, as the line endings might be the only
        // difference.
        Ok(Some(to_proto::text_edit_vec(
            &line_index,
            TextEdit::replace(TextRange::up_to(TextSize::of(&*file)), new_text),
        )))
    } else if *file == new_text {
        // The document is already formatted correctly -- no edits needed.
        Ok(None)
    } else {
        Ok(Some(to_proto::text_edit_vec(&line_index, diff(&file, &new_text))))
    }
}

pub(crate) fn fetch_dependency_list(
    state: GlobalStateSnapshot,
    _params: FetchDependencyListParams,
) -> anyhow::Result<FetchDependencyListResult> {
    let crates = state.analysis.fetch_crates()?;
    let crate_infos = crates
        .into_iter()
        .filter_map(|it| {
            let root_file_path = state.file_id_to_file_path(it.root_file_id);
            crate_path(&root_file_path).and_then(to_url).map(|path| CrateInfoResult {
                name: it.name,
                version: it.version,
                path,
            })
        })
        .collect();
    Ok(FetchDependencyListResult { crates: crate_infos })
}

pub(crate) fn internal_testing_fetch_config(
    state: GlobalStateSnapshot,
    params: InternalTestingFetchConfigParams,
) -> anyhow::Result<Option<InternalTestingFetchConfigResponse>> {
    let source_root = match params.text_document {
        Some(it) => Some(
            state
                .analysis
                .source_root_id(try_default!(from_proto::file_id(&state, &it.uri)?))
                .map_err(anyhow::Error::from)?,
        ),
        None => None,
    };
    Ok(Some(match params.config {
        InternalTestingFetchConfigOption::AssistEmitMustUse => {
            InternalTestingFetchConfigResponse::AssistEmitMustUse(
                state.config.assist(source_root).assist_emit_must_use,
            )
        }
        InternalTestingFetchConfigOption::CheckWorkspace => {
            InternalTestingFetchConfigResponse::CheckWorkspace(
                state.config.flycheck_workspace(source_root),
            )
        }
    }))
}

/// Searches for the directory of a Rust crate given this crate's root file path.
///
/// # Arguments
///
/// * `root_file_path`: The path to the root file of the crate.
///
/// # Returns
///
/// An `Option` value representing the path to the directory of the crate with the given
/// name, if such a crate is found. If no crate with the given name is found, this function
/// returns `None`.
fn crate_path(root_file_path: &VfsPath) -> Option<VfsPath> {
    let mut current_dir = root_file_path.parent();
    while let Some(path) = current_dir {
        let cargo_toml_path = path.join("../Cargo.toml")?;
        if fs::metadata(cargo_toml_path.as_path()?).is_ok() {
            let crate_path = cargo_toml_path.parent()?;
            return Some(crate_path);
        }
        current_dir = path.parent();
    }
    None
}

fn to_url(path: VfsPath) -> Option<Url> {
    let path = path.as_path()?;
    let str_path = path.as_os_str().to_str()?;
    Url::from_file_path(str_path).ok()
}

fn resource_ops_supported(config: &Config, kind: ResourceOperationKind) -> anyhow::Result<()> {
    if !matches!(config.workspace_edit_resource_operations(), Some(resops) if resops.contains(&kind))
    {
        return Err(LspError::new(
            ErrorCode::RequestFailed as i32,
            format!(
                "Client does not support {} capability.",
                match kind {
                    ResourceOperationKind::Create => "create",
                    ResourceOperationKind::Rename => "rename",
                    ResourceOperationKind::Delete => "delete",
                }
            ),
        )
        .into());
    }

    Ok(())
}

fn resolve_resource_op(op: &ResourceOp) -> ResourceOperationKind {
    match op {
        ResourceOp::Create(_) => ResourceOperationKind::Create,
        ResourceOp::Rename(_) => ResourceOperationKind::Rename,
        ResourceOp::Delete(_) => ResourceOperationKind::Delete,
    }
}

pub(crate) fn diff(left: &str, right: &str) -> TextEdit {
    use dissimilar::Chunk;

    let chunks = dissimilar::diff(left, right);

    let mut builder = TextEdit::builder();
    let mut pos = TextSize::default();

    let mut chunks = chunks.into_iter().peekable();
    while let Some(chunk) = chunks.next() {
        if let (Chunk::Delete(deleted), Some(&Chunk::Insert(inserted))) = (chunk, chunks.peek()) {
            chunks.next().unwrap();
            let deleted_len = TextSize::of(deleted);
            builder.replace(TextRange::at(pos, deleted_len), inserted.into());
            pos += deleted_len;
            continue;
        }

        match chunk {
            Chunk::Equal(text) => {
                pos += TextSize::of(text);
            }
            Chunk::Delete(deleted) => {
                let deleted_len = TextSize::of(deleted);
                builder.delete(TextRange::at(pos, deleted_len));
                pos += deleted_len;
            }
            Chunk::Insert(inserted) => {
                builder.insert(pos, inserted.into());
            }
        }
    }
    builder.finish()
}

#[test]
fn diff_smoke_test() {
    let mut original = String::from("fn foo(a:u32){\n}");
    let result = "fn foo(a: u32) {}";
    let edit = diff(&original, result);
    edit.apply(&mut original);
    assert_eq!(original, result);
}
