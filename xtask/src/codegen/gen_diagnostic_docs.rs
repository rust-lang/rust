//! Generates `assists.md` documentation.

use std::{fmt, path::PathBuf};

use xshell::write_file;

use crate::{
    codegen::{extract_comment_blocks_with_empty_lines, Location, PREAMBLE},
    project_root, rust_files, Result,
};

pub(crate) fn generate_diagnostic_docs() -> Result<()> {
    let diagnostics = Diagnostic::collect()?;
    let contents =
        diagnostics.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
    let contents = format!("//{}\n{}\n", PREAMBLE, contents.trim());
    let dst = project_root().join("docs/user/generated_diagnostic.adoc");
    write_file(&dst, &contents)?;
    Ok(())
}

#[derive(Debug)]
struct Diagnostic {
    id: String,
    location: Location,
    doc: String,
}

impl Diagnostic {
    fn collect() -> Result<Vec<Diagnostic>> {
        let mut res = Vec::new();
        for path in rust_files() {
            collect_file(&mut res, path)?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Diagnostic>, path: PathBuf) -> Result<()> {
            let text = xshell::read_file(&path)?;
            let comment_blocks = extract_comment_blocks_with_empty_lines("Diagnostic", &text);

            for block in comment_blocks {
                let id = block.id;
                if let Err(msg) = is_valid_diagnostic_name(&id) {
                    panic!("invalid diagnostic name: {:?}:\n  {}", id, msg)
                }
                let doc = block.contents.join("\n");
                let location = Location::new(path.clone(), block.line);
                acc.push(Diagnostic { id, location, doc })
            }

            Ok(())
        }
    }
}

fn is_valid_diagnostic_name(diagnostic: &str) -> Result<(), String> {
    let diagnostic = diagnostic.trim();
    if diagnostic.find(char::is_whitespace).is_some() {
        return Err("Diagnostic names can't contain whitespace symbols".into());
    }
    if diagnostic.chars().any(|c| c.is_ascii_uppercase()) {
        return Err("Diagnostic names can't contain uppercase symbols".into());
    }
    if diagnostic.chars().any(|c| !c.is_ascii()) {
        return Err("Diagnostic can't contain non-ASCII symbols".into());
    }

    Ok(())
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {}\n**Source:** {}\n{}", self.id, self.location, self.doc)
    }
}
