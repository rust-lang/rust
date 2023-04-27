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
    _params: lsp_ext::FetchDependencyListParams,
) -> Result<lsp_ext::FetchDependencyListResult> {
    let crates = state.analysis.fetch_crates()?;
    let crate_infos = crates
        .into_iter()
        .filter_map(|it| {
            let root_file_path = state.file_id_to_file_path(it.root_file_id);
            crate_path(root_file_path).and_then(to_url).map(|path| CrateInfoResult {
                name: it.name,
                version: it.version,
                path,
            })
        })
        .collect();
    Ok(FetchDependencyListResult { crates: crate_infos })
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
fn crate_path(root_file_path: VfsPath) -> Option<VfsPath> {
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