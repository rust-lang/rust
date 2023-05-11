//! Generates descriptors structure for unstable feature from Unstable Book
use std::{borrow::Cow, fs, path::Path};

use itertools::Itertools;
use stdx::format_to;
use test_utils::project_root;
use xshell::{cmd, Shell};

/// This clones rustc repo, and so is not worth to keep up-to-date. We update
/// manually by un-ignoring the test from time to time.
#[test]
#[ignore]
fn sourcegen_lint_completions() {
    let sh = &Shell::new().unwrap();

    let rust_repo = project_root().join("./target/rust");
    if !rust_repo.exists() {
        cmd!(sh, "git clone --depth=1 https://github.com/rust-lang/rust {rust_repo}")
            .run()
            .unwrap();
    }

    let mut contents = String::from(
        r"
#[derive(Clone)]
pub struct Lint {
    pub label: &'static str,
    pub description: &'static str,
}
pub struct LintGroup {
    pub lint: Lint,
    pub children: &'static [&'static str],
}
",
    );

    generate_lint_descriptor(sh, &mut contents);
    contents.push('\n');

    generate_feature_descriptor(&mut contents, &rust_repo.join("src/doc/unstable-book/src"));
    contents.push('\n');

    let lints_json = project_root().join("./target/clippy_lints.json");
    cmd!(
        sh,
        "curl https://rust-lang.github.io/rust-clippy/master/lints.json --output {lints_json}"
    )
    .run()
    .unwrap();
    generate_descriptor_clippy(&mut contents, &lints_json);

    let contents = sourcegen::add_preamble("sourcegen_lints", sourcegen::reformat(contents));

    let destination = project_root().join("crates/ide_db/src/generated/lints.rs");
    sourcegen::ensure_file_contents(destination.as_path(), &contents);
}

fn generate_lint_descriptor(sh: &Shell, buf: &mut String) {
    // FIXME: rustdoc currently requires an input file for -Whelp cc https://github.com/rust-lang/rust/pull/88831
    let file = project_root().join(file!());
    let stdout = cmd!(sh, "rustdoc -W help {file}").read().unwrap();
    let start_lints = stdout.find("----  -------  -------").unwrap();
    let start_lint_groups = stdout.find("----  ---------").unwrap();
    let start_lints_rustdoc =
        stdout.find("Lint checks provided by plugins loaded by this crate:").unwrap();
    let start_lint_groups_rustdoc =
        stdout.find("Lint groups provided by plugins loaded by this crate:").unwrap();

    buf.push_str(r#"pub const DEFAULT_LINTS: &[Lint] = &["#);
    buf.push('\n');

    let lints = stdout[start_lints..].lines().skip(1).take_while(|l| !l.is_empty()).map(|line| {
        let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
        let (_default_level, description) = rest.trim().split_once(char::is_whitespace).unwrap();
        (name.trim(), Cow::Borrowed(description.trim()), vec![])
    });
    let lint_groups =
        stdout[start_lint_groups..].lines().skip(1).take_while(|l| !l.is_empty()).map(|line| {
            let (name, lints) = line.trim().split_once(char::is_whitespace).unwrap();
            (
                name.trim(),
                format!("lint group for: {}", lints.trim()).into(),
                lints
                    .split_ascii_whitespace()
                    .map(|s| s.trim().trim_matches(',').replace('-', "_"))
                    .collect(),
            )
        });

    let lints = lints
        .chain(lint_groups)
        .sorted_by(|(ident, ..), (ident2, ..)| ident.cmp(ident2))
        .collect::<Vec<_>>();
    for (name, description, ..) in &lints {
        push_lint_completion(buf, &name.replace('-', "_"), description);
    }
    buf.push_str("];\n");
    buf.push_str(r#"pub const DEFAULT_LINT_GROUPS: &[LintGroup] = &["#);
    for (name, description, children) in &lints {
        if !children.is_empty() {
            // HACK: warnings is emitted with a general description, not with its members
            if name == &"warnings" {
                push_lint_group(buf, name, description, &Vec::new());
                continue;
            }
            push_lint_group(buf, &name.replace('-', "_"), description, children);
        }
    }
    buf.push('\n');
    buf.push_str("];\n");

    // rustdoc

    buf.push('\n');
    buf.push_str(r#"pub const RUSTDOC_LINTS: &[Lint] = &["#);
    buf.push('\n');

    let lints_rustdoc =
        stdout[start_lints_rustdoc..].lines().skip(2).take_while(|l| !l.is_empty()).map(|line| {
            let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
            let (_default_level, description) =
                rest.trim().split_once(char::is_whitespace).unwrap();
            (name.trim(), Cow::Borrowed(description.trim()), vec![])
        });
    let lint_groups_rustdoc =
        stdout[start_lint_groups_rustdoc..].lines().skip(2).take_while(|l| !l.is_empty()).map(
            |line| {
                let (name, lints) = line.trim().split_once(char::is_whitespace).unwrap();
                (
                    name.trim(),
                    format!("lint group for: {}", lints.trim()).into(),
                    lints
                        .split_ascii_whitespace()
                        .map(|s| s.trim().trim_matches(',').replace('-', "_"))
                        .collect(),
                )
            },
        );

    let lints_rustdoc = lints_rustdoc
        .chain(lint_groups_rustdoc)
        .sorted_by(|(ident, ..), (ident2, ..)| ident.cmp(ident2))
        .collect::<Vec<_>>();

    for (name, description, ..) in &lints_rustdoc {
        push_lint_completion(buf, &name.replace('-', "_"), description)
    }
    buf.push_str("];\n");

    buf.push_str(r#"pub const RUSTDOC_LINT_GROUPS: &[LintGroup] = &["#);
    for (name, description, children) in &lints_rustdoc {
        if !children.is_empty() {
            push_lint_group(buf, &name.replace('-', "_"), description, children);
        }
    }
    buf.push('\n');
    buf.push_str("];\n");
}

fn generate_feature_descriptor(buf: &mut String, src_dir: &Path) {
    let mut features = ["language-features", "library-features"]
        .into_iter()
        .flat_map(|it| sourcegen::list_files(&src_dir.join(it)))
        .filter(|path| {
            // Get all `.md ` files
            path.extension().unwrap_or_default().to_str().unwrap_or_default() == "md"
        })
        .map(|path| {
            let feature_ident = path.file_stem().unwrap().to_str().unwrap().replace('-', "_");
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
    let mut clippy_groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();

    for line in file_content.lines().map(|line| line.trim()) {
        if let Some(line) = line.strip_prefix(r#""id": ""#) {
            let clippy_lint = ClippyLint {
                id: line.strip_suffix(r#"","#).expect("should be suffixed by comma").into(),
                help: String::new(),
            };
            clippy_lints.push(clippy_lint)
        } else if let Some(line) = line.strip_prefix(r#""group": ""#) {
            if let Some(group) = line.strip_suffix("\",") {
                clippy_groups
                    .entry(group.to_owned())
                    .or_default()
                    .push(clippy_lints.last().unwrap().id.clone());
            }
        } else if let Some(line) = line.strip_prefix(r#""docs": ""#) {
            let prefix_to_strip = r#" ### What it does"#;
            let line = match line.strip_prefix(prefix_to_strip) {
                Some(line) => line,
                None => {
                    eprintln!("unexpected clippy prefix for {}", clippy_lints.last().unwrap().id);
                    continue;
                }
            };
            // Only take the description, any more than this is a lot of additional data we would embed into the exe
            // which seems unnecessary
            let up_to = line.find(r#"###"#).expect("no second section found?");
            let line = &line[..up_to];

            let clippy_lint = clippy_lints.last_mut().expect("clippy lint must already exist");
            clippy_lint.help = unescape(line).trim().to_string();
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

    buf.push_str(r#"pub const CLIPPY_LINT_GROUPS: &[LintGroup] = &["#);
    for (id, children) in clippy_groups {
        let children = children.iter().map(|id| format!("clippy::{id}")).collect::<Vec<_>>();
        if !children.is_empty() {
            let lint_ident = format!("clippy::{id}");
            let description = format!("lint group for: {}", children.iter().join(", "));
            push_lint_group(buf, &lint_ident, &description, &children);
        }
    }
    buf.push('\n');
    buf.push_str("];\n");
}

fn push_lint_completion(buf: &mut String, label: &str, description: &str) {
    format_to!(
        buf,
        r###"    Lint {{
        label: "{}",
        description: r##"{}"##,
    }},"###,
        label,
        description,
    );
}

fn push_lint_group(buf: &mut String, label: &str, description: &str, children: &[String]) {
    buf.push_str(
        r###"    LintGroup {
        lint:
        "###,
    );

    push_lint_completion(buf, label, description);

    let children = format!("&[{}]", children.iter().map(|it| format!("\"{it}\"")).join(", "));
    format_to!(
        buf,
        r###"
        children: {},
        }},"###,
        children,
    );
}
