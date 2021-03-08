//! Generates descriptors structure for unstable feature from Unstable Book
use std::fmt::Write;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;
use xshell::{cmd, read_file};

use crate::codegen::{ensure_file_contents, project_root, reformat, Result};

pub(crate) fn generate_lint_completions() -> Result<()> {
    if !Path::new("./target/rust").exists() {
        cmd!("git clone --depth=1 https://github.com/rust-lang/rust ./target/rust").run()?;
    }

    let mut contents = String::from("use crate::completions::attribute::LintCompletion;\n\n");
    generate_descriptor(&mut contents, "./target/rust/src/doc/unstable-book/src".into())?;
    contents.push('\n');

    cmd!("curl http://rust-lang.github.io/rust-clippy/master/lints.json --output ./target/clippy_lints.json").run()?;
    generate_descriptor_clippy(&mut contents, &Path::new("./target/clippy_lints.json"))?;
    let contents = reformat(&contents)?;

    let destination =
        project_root().join("crates/ide_completion/src/generated_lint_completions.rs");
    ensure_file_contents(destination.as_path(), &contents)?;

    Ok(())
}

fn generate_descriptor(buf: &mut String, src_dir: PathBuf) -> Result<()> {
    buf.push_str(r#"pub(super) const FEATURES: &[LintCompletion] = &["#);
    buf.push('\n');
    ["language-features", "library-features"]
        .iter()
        .flat_map(|it| WalkDir::new(src_dir.join(it)))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file() && entry.path().extension().unwrap_or_default() == "md"
        })
        .for_each(|entry| {
            let path = entry.path();
            let feature_ident = path.file_stem().unwrap().to_str().unwrap().replace("-", "_");
            let doc = read_file(path).unwrap();

            push_lint_completion(buf, &feature_ident, &doc);
        });
    buf.push_str("];\n");
    Ok(())
}

#[derive(Default)]
struct ClippyLint {
    help: String,
    id: String,
}

fn generate_descriptor_clippy(buf: &mut String, path: &Path) -> Result<()> {
    let file_content = read_file(path)?;
    let mut clippy_lints: Vec<ClippyLint> = vec![];

    for line in file_content.lines().map(|line| line.trim()) {
        if line.starts_with(r#""id":"#) {
            let clippy_lint = ClippyLint {
                id: line
                    .strip_prefix(r#""id": ""#)
                    .expect("should be prefixed by id")
                    .strip_suffix(r#"","#)
                    .expect("should be suffixed by comma")
                    .into(),
                help: String::new(),
            };
            clippy_lints.push(clippy_lint)
        } else if line.starts_with(r#""What it does":"#) {
            // Typical line to strip: "What is doest": "Here is my useful content",
            let prefix_to_strip = r#""What it does": ""#;
            let suffix_to_strip = r#"","#;

            let clippy_lint = clippy_lints.last_mut().expect("clippy lint must already exist");
            clippy_lint.help = line
                .strip_prefix(prefix_to_strip)
                .expect("should be prefixed by what it does")
                .strip_suffix(suffix_to_strip)
                .expect("should be suffixed by comma")
                .into();
        }
    }

    buf.push_str(r#"pub(super) const CLIPPY_LINTS: &[LintCompletion] = &["#);
    buf.push('\n');
    clippy_lints.into_iter().for_each(|clippy_lint| {
        let lint_ident = format!("clippy::{}", clippy_lint.id);
        let doc = clippy_lint.help;
        push_lint_completion(buf, &lint_ident, &doc);
    });

    buf.push_str("];\n");

    Ok(())
}

fn push_lint_completion(buf: &mut String, label: &str, description: &str) {
    writeln!(
        buf,
        r###"    LintCompletion {{
        label: "{}",
        description: r##"{}"##
    }},"###,
        label, description
    )
    .unwrap();
}
