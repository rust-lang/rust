use std::path::{Path, PathBuf};
use std::{fs, str};

use rustc_errors::DiagCtxtHandle;
use rustc_span::edition::Edition;
use serde::Serialize;

use crate::html::markdown::{ErrorCodes, HeadingOffset, IdMap, Markdown, Playground};

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
        dcx: DiagCtxtHandle<'_>,
        id_map: &mut IdMap,
        edition: Edition,
        playground: &Option<Playground>,
        loaded_paths: &mut Vec<PathBuf>,
    ) -> Option<ExternalHtml> {
        let codes = ErrorCodes::from(nightly_build);
        let ih = load_external_files(in_header, dcx, loaded_paths)?;
        let bc = {
            let mut bc = load_external_files(before_content, dcx, loaded_paths)?;
            let m_bc = load_external_files(md_before_content, dcx, loaded_paths)?;
            Markdown {
                content: &m_bc,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
            }
            .write_into(&mut bc)
            .unwrap();
            bc
        };
        let ac = {
            let mut ac = load_external_files(after_content, dcx, loaded_paths)?;
            let m_ac = load_external_files(md_after_content, dcx, loaded_paths)?;
            Markdown {
                content: &m_ac,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
            }
            .write_into(&mut ac)
            .unwrap();
            ac
        };
        Some(ExternalHtml { in_header: ih, before_content: bc, after_content: ac })
    }
}

pub(crate) enum LoadStringError {
    ReadFail,
    BadUtf8,
}

pub(crate) fn load_string<P: AsRef<Path>>(
    file_path: P,
    dcx: DiagCtxtHandle<'_>,
    loaded_paths: &mut Vec<PathBuf>,
) -> Result<String, LoadStringError> {
    let file_path = file_path.as_ref();
    loaded_paths.push(file_path.to_owned());
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

fn load_external_files(
    names: &[String],
    dcx: DiagCtxtHandle<'_>,
    loaded_paths: &mut Vec<PathBuf>,
) -> Option<String> {
    let mut out = String::new();
    for name in names {
        let Ok(s) = load_string(name, dcx, loaded_paths) else { return None };
        out.push_str(&s);
        out.push('\n');
    }
    Some(out)
}
