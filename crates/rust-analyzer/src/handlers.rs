//! This module is responsible for implementing handlers for Language Server
//! Protocol. The majority of requests are fulfilled by calling into the
//! `ide` crate.

use ide::AssistResolveStrategy;
use lsp_types::{Diagnostic, DiagnosticTag, NumberOrString};
use vfs::FileId;

use crate::{global_state::GlobalStateSnapshot, to_proto, Result};

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
