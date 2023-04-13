//! This module is responsible for implementing handlers for Language Server
//! Protocol. The majority of requests are fulfilled by calling into the
//! `ide` crate.

use ide::AssistResolveStrategy;
use lsp_types::{Diagnostic, DiagnosticTag, NumberOrString};
use vfs::FileId;

use crate::{
    global_state::GlobalStateSnapshot, to_proto, Result,
    lsp_ext::{
        CrateInfoResult, FetchDependencyListResult, FetchDependencyListParams,
    },
};


pub(crate) mod request;
pub(crate) mod notification;

pub(crate) fn publish_diagnostics(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
) -> Result<Vec<Diagnostic>> {
    let _p = profile::span("publish_diagnostics");
    let line_index = snap.file_line_index(file_id)?;

    let diagnostics: Vec<Diagnostic> = snap
        .analysis
        .diagnostics(&snap.config.diagnostics(), AssistResolveStrategy::None, file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: to_proto::range(&line_index, d.range),
            severity: Some(to_proto::diagnostic_severity(d.severity)),
            code: Some(NumberOrString::String(d.code.as_str().to_string())),
            code_description: Some(lsp_types::CodeDescription {
                href: lsp_types::Url::parse(&format!(
                    "https://rust-analyzer.github.io/manual.html#{}",
                    d.code.as_str()
                ))
                    .unwrap(),
            }),
            source: Some("rust-analyzer".to_string()),
            message: d.message,
            related_information: None,
            tags: if d.unused { Some(vec![DiagnosticTag::UNNECESSARY]) } else { None },
            data: None,
        })
        .collect();
    Ok(diagnostics)
}

pub(crate) fn fetch_dependency_list(
    state: GlobalStateSnapshot,
    _params: FetchDependencyListParams,
) -> Result<FetchDependencyListResult> {
    let crates = state.analysis.fetch_crates()?;
    Ok(FetchDependencyListResult {
        crates: crates
            .into_iter()
            .filter_map(|it| {
                let root_file_path = state.file_id_to_file_path(it.root_file_id);
                crate_path(it.name.as_ref(), root_file_path).map(|crate_path| CrateInfoResult {
                    name: it.name,
                    version: it.version,
                    path: crate_path.to_string(),
                })
            })
            .collect(),
    })
}

//Thats a best effort to try and find the crate path
fn crate_path(crate_name: Option<&String>, root_file_path: VfsPath) -> Option<VfsPath> {
    crate_name.and_then(|crate_name| {
        let mut crate_path = None;
        let mut root_path = root_file_path;
        while let Some(path) = root_path.parent() {
            match path.name_and_extension() {
                Some((name, _)) => {
                    if name.starts_with(crate_name.as_str()) {
                        crate_path = Some(path);
                        break;
                    }
                }
                None => break,
            }
            root_path = path;
        }
        crate_path
    })
}