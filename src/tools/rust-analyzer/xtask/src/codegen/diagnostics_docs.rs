//! Generates `assists.md` documentation.

use std::{fmt, fs, io, path::PathBuf};

use crate::{
    codegen::{add_preamble, CommentBlock, Location},
    project_root,
    util::list_rust_files,
};

pub(crate) fn generate(check: bool) {
    let diagnostics = Diagnostic::collect().unwrap();
    if !check {
        let contents =
            diagnostics.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
        let contents = add_preamble(crate::flags::CodegenType::DiagnosticsDocs, contents);
        let dst = project_root().join("docs/user/generated_diagnostic.adoc");
        fs::write(dst, contents).unwrap();
    }
}

#[derive(Debug)]
struct Diagnostic {
    id: String,
    location: Location,
    doc: String,
}

impl Diagnostic {
    fn collect() -> io::Result<Vec<Diagnostic>> {
        let handlers_dir = project_root().join("crates/ide-diagnostics/src/handlers");

        let mut res = Vec::new();
        for path in list_rust_files(&handlers_dir) {
            collect_file(&mut res, path)?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Diagnostic>, path: PathBuf) -> io::Result<()> {
            let text = fs::read_to_string(&path)?;
            let comment_blocks = CommentBlock::extract("Diagnostic", &text);

            for block in comment_blocks {
                let id = block.id;
                if let Err(msg) = is_valid_diagnostic_name(&id) {
                    panic!("invalid diagnostic name: {id:?}:\n  {msg}")
                }
                let doc = block.contents.join("\n");
                let location = Location { file: path.clone(), line: block.line };
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
    if !diagnostic.is_ascii() {
        return Err("Diagnostic can't contain non-ASCII symbols".into());
    }

    Ok(())
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {}\n**Source:** {}\n{}", self.id, self.location, self.doc)
    }
}
