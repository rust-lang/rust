//! Generates descriptors structure for unstable feature from Unstable Book
use std::borrow::Cow;
use std::fmt::Write;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;
use xshell::{cmd, read_file};

use crate::codegen::{ensure_file_contents, project_root, reformat, Result};

pub(crate) fn generate_lint_completions() -> Result<()> {
    if !project_root().join("./target/rust").exists() {
        cmd!("git clone --depth=1 https://github.com/rust-lang/rust ./target/rust").run()?;
    }

    let mut contents = String::from(
        r#"pub struct Lint {
    pub label: &'static str,
    pub description: &'static str,
}

"#,
    );
    generate_lint_descriptor(&mut contents)?;
    contents.push('\n');

    generate_feature_descriptor(&mut contents, "./target/rust/src/doc/unstable-book/src".into())?;
    contents.push('\n');

    cmd!("curl http://rust-lang.github.io/rust-clippy/master/lints.json --output ./target/clippy_lints.json").run()?;
    generate_descriptor_clippy(&mut contents, &Path::new("./target/clippy_lints.json"))?;
    let contents = reformat(&contents)?;

    let destination = project_root().join("crates/ide_db/src/helpers/generated_lints.rs");
    ensure_file_contents(destination.as_path(), &contents)?;

    Ok(())
}

fn generate_lint_descriptor(buf: &mut String) -> Result<()> {
    let stdout = cmd!("rustc -W help").read()?;
    let start_lints =
        stdout.find("----  -------  -------").ok_or_else(|| anyhow::format_err!(""))?;
    let start_lint_groups =
        stdout.find("----  ---------").ok_or_else(|| anyhow::format_err!(""))?;
    let end_lints =
        stdout.find("Lint groups provided by rustc:").ok_or_else(|| anyhow::format_err!(""))?;
    let end_lint_groups = stdout
        .find("Lint tools like Clippy can provide additional lints and lint groups.")
        .ok_or_else(|| anyhow::format_err!(""))?;
    buf.push_str(r#"pub const DEFAULT_LINTS: &[Lint] = &["#);
    buf.push('\n');
    let mut lints = stdout[start_lints..end_lints]
        .lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
            let (_default_level, description) =
                rest.trim().split_once(char::is_whitespace).unwrap();
            (name.trim(), Cow::Borrowed(description.trim()))
        })
        .collect::<Vec<_>>();
    lints.extend(
        stdout[start_lint_groups..end_lint_groups].lines().skip(1).filter(|l| !l.is_empty()).map(
            |line| {
                let (name, lints) = line.trim().split_once(char::is_whitespace).unwrap();
                (name.trim(), format!("lint group for: {}", lints.trim()).into())
            },
        ),
    );

    lints.sort_by(|(ident, _), (ident2, _)| ident.cmp(ident2));
    lints.into_iter().for_each(|(name, description)| {
        push_lint_completion(buf, &name.replace("-", "_"), &description)
    });
    buf.push_str("];\n");
    Ok(())
}

fn generate_feature_descriptor(buf: &mut String, src_dir: PathBuf) -> Result<()> {
    buf.push_str(r#"pub const FEATURES: &[Lint] = &["#);
    buf.push('\n');
    let mut vec = ["language-features", "library-features"]
        .iter()
        .flat_map(|it| WalkDir::new(src_dir.join(it)))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file() && entry.path().extension().unwrap_or_default() == "md"
        })
        .map(|entry| {
            let path = entry.path();
            let feature_ident = path.file_stem().unwrap().to_str().unwrap().replace("-", "_");
            let doc = read_file(path).unwrap();
            (feature_ident, doc)
        })
        .collect::<Vec<_>>();
    vec.sort_by(|(feature_ident, _), (feature_ident2, _)| feature_ident.cmp(feature_ident2));
    vec.into_iter()
        .for_each(|(feature_ident, doc)| push_lint_completion(buf, &feature_ident, &doc));
    buf.push_str("];\n");
    Ok(())
}

#[derive(Default)]
struct ClippyLint {
    help: String,
    id: String,
}

fn unescape(s: &str) -> String {
    s.replace(r#"\""#, "").replace(r#"\n"#, "\n").replace(r#"\r"#, "")
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
                .map(unescape)
                .expect("should be suffixed by comma")
                .into();
        }
    }
    clippy_lints.sort_by(|lint, lint2| lint.id.cmp(&lint2.id));
    buf.push_str(r#"pub const CLIPPY_LINTS: &[Lint] = &["#);
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
        r###"    Lint {{
        label: "{}",
        description: r##"{}"##
    }},"###,
        label, description
    )
    .unwrap();
}
