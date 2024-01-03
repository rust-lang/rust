use crate::html::markdown::{ErrorCodes, HeadingOffset, IdMap, Markdown, Playground};
use crate::rustc_span::edition::Edition;
use std::fs;
use std::path::Path;
use std::str;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ExternalHtml {
    /// Content that will be included inline in the `<head>` section of a
    /// rendered Markdown file or generated documentation
    pub(crate) in_header: String,
    /// Content that will be included inline between `<body>` and the content of
    /// a rendered Markdown file or generated documentation
    pub(crate) before_content: String,
    /// Content that will be included inline between the content and `</body>` of
    /// a rendered Markdown file or generated documentation
    pub(crate) after_content: String,
}

impl ExternalHtml {
    pub(crate) fn load(
        in_header: &[String],
        before_content: &[String],
        after_content: &[String],
        md_before_content: &[String],
        md_after_content: &[String],
        nightly_build: bool,
        dcx: &rustc_errors::DiagCtxt,
        id_map: &mut IdMap,
        edition: Edition,
        playground: &Option<Playground>,
    ) -> Option<ExternalHtml> {
        let codes = ErrorCodes::from(nightly_build);
        let ih = load_external_files(in_header, dcx)?;
        let bc = load_external_files(before_content, dcx)?;
        let m_bc = load_external_files(md_before_content, dcx)?;
        let bc = format!(
            "{bc}{}",
            Markdown {
                content: &m_bc,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
                // For external files, it'll be disabled until the feature is enabled by default.
                custom_code_classes_in_docs: false,
            }
            .into_string()
        );
        let ac = load_external_files(after_content, dcx)?;
        let m_ac = load_external_files(md_after_content, dcx)?;
        let ac = format!(
            "{ac}{}",
            Markdown {
                content: &m_ac,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
                // For external files, it'll be disabled until the feature is enabled by default.
                custom_code_classes_in_docs: false,
            }
            .into_string()
        );
        Some(ExternalHtml { in_header: ih, before_content: bc, after_content: ac })
    }
}

pub(crate) enum LoadStringError {
    ReadFail,
    BadUtf8,
}

pub(crate) fn load_string<P: AsRef<Path>>(
    file_path: P,
    dcx: &rustc_errors::DiagCtxt,
) -> Result<String, LoadStringError> {
    let file_path = file_path.as_ref();
    let contents = match fs::read(file_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            dcx.struct_err(format!(
                "error reading `{file_path}`: {e}",
                file_path = file_path.display()
            ))
            .emit();
            return Err(LoadStringError::ReadFail);
        }
    };
    match str::from_utf8(&contents) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            dcx.err(format!("error reading `{}`: not UTF-8", file_path.display()));
            Err(LoadStringError::BadUtf8)
        }
    }
}

fn load_external_files(names: &[String], dcx: &rustc_errors::DiagCtxt) -> Option<String> {
    let mut out = String::new();
    for name in names {
        let Ok(s) = load_string(name, dcx) else { return None };
        out.push_str(&s);
        out.push('\n');
    }
    Some(out)
}
