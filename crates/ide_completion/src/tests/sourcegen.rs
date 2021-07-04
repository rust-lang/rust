//! Generates descriptors structure for unstable feature from Unstable Book
use std::{borrow::Cow, fs, path::Path};

use stdx::format_to;
use test_utils::project_root;
use xshell::cmd;

/// This clones rustc repo, and so is not worth to keep up-to-date. We update
/// manually by un-ignoring the test from time to time.
#[test]
#[ignore]
fn sourcegen_lint_completions() {
    let rust_repo = project_root().join("./target/rust");
    if !rust_repo.exists() {
        cmd!("git clone --depth=1 https://github.com/rust-lang/rust {rust_repo}").run().unwrap();
    }

    let mut contents = r"
pub struct Lint {
    pub label: &'static str,
    pub description: &'static str,
}
"
    .to_string();
    generate_lint_descriptor(&mut contents);
    contents.push('\n');

    generate_feature_descriptor(&mut contents, &rust_repo.join("src/doc/unstable-book/src"));
    contents.push('\n');

    let lints_json = project_root().join("./target/clippy_lints.json");
    cmd!("curl https://rust-lang.github.io/rust-clippy/master/lints.json --output {lints_json}")
        .run()
        .unwrap();
    generate_descriptor_clippy(&mut contents, &lints_json);

    let contents =
        sourcegen::add_preamble("sourcegen_lint_completions", sourcegen::reformat(contents));

    let destination = project_root().join("crates/ide_db/src/helpers/generated_lints.rs");
    sourcegen::ensure_file_contents(destination.as_path(), &contents);
}

fn generate_lint_descriptor(buf: &mut String) {
    let stdout = cmd!("rustc -W help").read().unwrap();
    let start_lints = stdout.find("----  -------  -------").unwrap();
    let start_lint_groups = stdout.find("----  ---------").unwrap();
    let end_lints = stdout.find("Lint groups provided by rustc:").unwrap();
    let end_lint_groups = stdout
        .find("Lint tools like Clippy can provide additional lints and lint groups.")
        .unwrap();
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
}

fn generate_feature_descriptor(buf: &mut String, src_dir: &Path) {
    let mut features = ["language-features", "library-features"]
        .iter()
        .flat_map(|it| sourcegen::list_files(&src_dir.join(it)))
        .filter(|path| {
            // Get all `.md ` files
            path.extension().unwrap_or_default().to_str().unwrap_or_default() == "md"
        })
        .map(|path| {
            let feature_ident = path.file_stem().unwrap().to_str().unwrap().replace("-", "_");
            let doc = fs::read_to_string(path).unwrap();
            (feature_ident, doc)
        })
        .collect::<Vec<_>>();
    features.sort_by(|(feature_ident, _), (feature_ident2, _)| feature_ident.cmp(feature_ident2));

    buf.push_str(r#"pub const FEATURES: &[Lint] = &["#);
    for (feature_ident, doc) in features.into_iter() {
        push_lint_completion(buf, &feature_ident, &doc)
    }
    buf.push('\n');
    buf.push_str("];\n");
}

#[derive(Default)]
struct ClippyLint {
    help: String,
    id: String,
}

fn unescape(s: &str) -> String {
    s.replace(r#"\""#, "").replace(r#"\n"#, "\n").replace(r#"\r"#, "")
}

fn generate_descriptor_clippy(buf: &mut String, path: &Path) {
    let file_content = std::fs::read_to_string(path).unwrap();
    let mut clippy_lints: Vec<ClippyLint> = Vec::new();

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
                .expect("should be suffixed by comma");
        }
    }
    clippy_lints.sort_by(|lint, lint2| lint.id.cmp(&lint2.id));

    buf.push_str(r#"pub const CLIPPY_LINTS: &[Lint] = &["#);
    buf.push('\n');
    for clippy_lint in clippy_lints.into_iter() {
        let lint_ident = format!("clippy::{}", clippy_lint.id);
        let doc = clippy_lint.help;
        push_lint_completion(buf, &lint_ident, &doc);
    }
    buf.push_str("];\n");
}

fn push_lint_completion(buf: &mut String, label: &str, description: &str) {
    format_to!(
        buf,
        r###"    Lint {{
        label: "{}",
        description: r##"{}"##
    }},"###,
        label,
        description
    );
}
