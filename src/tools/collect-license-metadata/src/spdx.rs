use anyhow::Error;

/// A single file entry extracted from an SPDX tag-value document.
pub(crate) struct SpdxFileEntry {
    pub(crate) name: String,
    pub(crate) concluded_license: String,
    pub(crate) copyright_text: String,
}

/// Parses an SPDX tag-value document and extracts file information.
///
/// This is a minimal parser that only extracts the fields we need
/// (FileName, LicenseConcluded, FileCopyrightText) rather than the full model.
/// The format is specified by the SPDX specification:
/// each line is a `Tag: Value` pair,
/// and multi-line values are wrapped in `<text>…</text>`.
pub(crate) fn parse_tag_value(input: &str) -> Result<Vec<SpdxFileEntry>, Error> {
    let mut files = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_license: Option<String> = None;
    let mut current_copyright: Option<String> = None;

    let mut lines = input.lines();
    while let Some(line) = lines.next() {
        let Some((tag, value)) = line.split_once(": ") else {
            continue;
        };

        let value = resolve_multiline_value(value, &mut lines)?;

        match tag {
            "FileName" => {
                // A new file section begins. Flush the previous one if present.
                if let Some(name) = current_name.take() {
                    files.push(build_file_entry(
                        name,
                        current_license.take(),
                        current_copyright.take(),
                    )?);
                }
                current_name = Some(value);
                current_license = None;
                current_copyright = None;
            }
            "LicenseConcluded" => current_license = Some(value),
            "FileCopyrightText" => current_copyright = Some(value),
            _ => {}
        }
    }

    // Flush the last file section.
    if let Some(name) = current_name {
        files.push(build_file_entry(name, current_license, current_copyright)?);
    }

    Ok(files)
}

/// Resolves a tag value that might span multiple lines using `<text>…</text>`.
fn resolve_multiline_value<'a>(
    value: &str,
    further_lines: &mut impl Iterator<Item = &'a str>,
) -> Result<String, Error> {
    let Some(start) = value.strip_prefix("<text>") else {
        return Ok(value.to_string());
    };

    // The closing tag might be on the same line.
    if let Some(content) = start.strip_suffix("</text>") {
        return Ok(content.to_string());
    }

    let mut text = start.to_string();
    for line in further_lines.by_ref() {
        if let Some(rest) = line.strip_suffix("</text>") {
            text.push('\n');
            text.push_str(rest);
            return Ok(text);
        }
        text.push('\n');
        text.push_str(line);
    }

    anyhow::bail!("unexpected end of input inside <text> block")
}

fn build_file_entry(
    name: String,
    concluded_license: Option<String>,
    copyright_text: Option<String>,
) -> Result<SpdxFileEntry, Error> {
    Ok(SpdxFileEntry {
        name,
        concluded_license: concluded_license
            .ok_or_else(|| anyhow::anyhow!("file missing LicenseConcluded"))?,
        copyright_text: copyright_text
            .ok_or_else(|| anyhow::anyhow!("file missing FileCopyrightText"))?,
    })
}
