use crate::html::markdown::{ErrorCodes, HeadingOffset, IdMap, Markdown, Playground};
use crate::rustc_span::edition::Edition;
use std::fs;
use std::path::Path;
use std::str;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
crate struct ExternalHtml {
    /// Content that will be included inline in the `<head>` section of a
    /// rendered Markdown file or generated documentation
    crate in_header: String,
    /// Content that will be included inline between `<body>` and the content of
    /// a rendered Markdown file or generated documentation
    crate before_content: String,
    /// Content that will be included inline between the content and `</body>` of
    /// a rendered Markdown file or generated documentation
    crate after_content: String,
}

impl ExternalHtml {
    crate fn load(
        in_header: &[String],
        before_content: &[String],
        after_content: &[String],
        md_before_content: &[String],
        md_after_content: &[String],
        nightly_build: bool,
        diag: &rustc_errors::Handler,
        id_map: &mut IdMap,
        edition: Edition,
        playground: &Option<Playground>,
    ) -> Option<ExternalHtml> {
        let codes = ErrorCodes::from(nightly_build);
        let ih = load_external_files(in_header, diag)?;
        let bc = load_external_files(before_content, diag)?;
        let m_bc = load_external_files(md_before_content, diag)?;
        let bc = format!(
            "{}{}",
            bc,
            Markdown {
                content: &m_bc,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
            }
            .into_string()
        );
        let ac = load_external_files(after_content, diag)?;
        let m_ac = load_external_files(md_after_content, diag)?;
        let ac = format!(
            "{}{}",
            ac,
            Markdown {
                content: &m_ac,
                links: &[],
                ids: id_map,
                error_codes: codes,
                edition,
                playground,
                heading_offset: HeadingOffset::H2,
            }
            .into_string()
        );
        Some(ExternalHtml { in_header: ih, before_content: bc, after_content: ac })
    }
}

crate enum LoadStringError {
    ReadFail,
    BadUtf8,
}

crate fn load_string<P: AsRef<Path>>(
    file_path: P,
    diag: &rustc_errors::Handler,
) -> Result<String, LoadStringError> {
    let file_path = file_path.as_ref();
    let contents = match fs::read(file_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            diag.struct_err(&format!("error reading `{}`: {}", file_path.display(), e)).emit();
            return Err(LoadStringError::ReadFail);
        }
    };
    match str::from_utf8(&contents) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            diag.struct_err(&format!("error reading `{}`: not UTF-8", file_path.display())).emit();
            Err(LoadStringError::BadUtf8)
        }
    }
}

fn load_external_files(names: &[String], diag: &rustc_errors::Handler) -> Option<String> {
    let mut out = String::new();
    for name in names {
        let s = match load_string(name, diag) {
            Ok(s) => s,
            Err(_) => return None,
        };
        out.push_str(&s);
        out.push('\n');
    }
    Some(out)
}
