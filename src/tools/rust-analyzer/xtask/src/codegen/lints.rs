//! Generates descriptor structures for unstable features from the unstable book
//! and lints from rustc, rustdoc, and clippy.
use std::{borrow::Cow, fs, path::Path};

use stdx::format_to;
use xshell::{cmd, Shell};

use crate::{
    codegen::{add_preamble, ensure_file_contents, list_files, reformat},
    project_root,
};

const DESTINATION: &str = "crates/ide-db/src/generated/lints.rs";

/// This clones rustc repo, and so is not worth to keep up-to-date on a constant basis.
pub(crate) fn generate(check: bool) {
    let sh = &Shell::new().unwrap();

    let rust_repo = project_root().join("./target/rust");
    if rust_repo.exists() {
        cmd!(sh, "git -C {rust_repo} pull --rebase").run().unwrap();
    } else {
        cmd!(sh, "git clone --depth=1 https://github.com/rust-lang/rust {rust_repo}")
            .run()
            .unwrap();
    }
    // need submodules for Cargo to parse the workspace correctly
    cmd!(
        sh,
        "git -C {rust_repo} submodule update --init --recursive --depth=1 --
         compiler library src/tools"
    )
    .run()
    .unwrap();

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

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let unstable_book = project_root().join("./target/unstable-book-gen");
    cmd!(
        sh,
        "{cargo} run --manifest-path {rust_repo}/src/tools/unstable-book-gen/Cargo.toml --
         {rust_repo}/library {rust_repo}/compiler {rust_repo}/src {unstable_book}"
    )
    .run()
    .unwrap();
    generate_feature_descriptor(&mut contents, &unstable_book.join("src"));
    contents.push('\n');

    let lints_json = project_root().join("./target/clippy_lints.json");
    cmd!(
        sh,
        "curl https://rust-lang.github.io/rust-clippy/master/lints.json --output {lints_json}"
    )
    .run()
    .unwrap();
    generate_descriptor_clippy(&mut contents, &lints_json);

    let contents = add_preamble("sourcegen_lints", reformat(contents));

    let destination = project_root().join(DESTINATION);
    ensure_file_contents(destination.as_path(), &contents, check);
}

/// Parses the output of `rustdoc -Whelp` and prints `Lint` and `LintGroup` constants into `buf`.
///
/// As of writing, the output of `rustc -Whelp` (not rustdoc) has the following format:
///
/// ```text
/// Lint checks provided by rustc:
///
/// name  default  meaning
/// ----  -------  -------
///
/// ...
///
/// Lint groups provided by rustc:
///
/// name  sub-lints
/// ----  ---------
///
/// ...
/// ```
///
/// `rustdoc -Whelp` (and any other custom `rustc` driver) adds another two
/// tables after the `rustc` ones, with a different title but the same format.
fn generate_lint_descriptor(sh: &Shell, buf: &mut String) {
    let stdout = cmd!(sh, "rustdoc -Whelp").read().unwrap();
    let lints_pat = "----  -------  -------\n";
    let lint_groups_pat = "----  ---------\n";
    let lints = find_and_slice(&stdout, lints_pat);
    let lint_groups = find_and_slice(lints, lint_groups_pat);
    let lints_rustdoc = find_and_slice(lint_groups, lints_pat);
    let lint_groups_rustdoc = find_and_slice(lints_rustdoc, lint_groups_pat);

    buf.push_str(r#"pub const DEFAULT_LINTS: &[Lint] = &["#);
    buf.push('\n');

    let lints = lints.lines().take_while(|l| !l.is_empty()).map(|line| {
        let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
        let (_default_level, description) = rest.trim().split_once(char::is_whitespace).unwrap();
        (name.trim(), Cow::Borrowed(description.trim()), vec![])
    });
    let lint_groups = lint_groups.lines().take_while(|l| !l.is_empty()).map(|line| {
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

    let mut lints = lints.chain(lint_groups).collect::<Vec<_>>();
    lints.sort_by(|(ident, ..), (ident2, ..)| ident.cmp(ident2));

    for (name, description, ..) in &lints {
        push_lint_completion(buf, &name.replace('-', "_"), description);
    }
    buf.push_str("];\n\n");

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

    let lints_rustdoc = lints_rustdoc.lines().take_while(|l| !l.is_empty()).map(|line| {
        let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
        let (_default_level, description) = rest.trim().split_once(char::is_whitespace).unwrap();
        (name.trim(), Cow::Borrowed(description.trim()), vec![])
    });
    let lint_groups_rustdoc =
        lint_groups_rustdoc.lines().take_while(|l| !l.is_empty()).map(|line| {
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

    let mut lints_rustdoc = lints_rustdoc.chain(lint_groups_rustdoc).collect::<Vec<_>>();
    lints_rustdoc.sort_by(|(ident, ..), (ident2, ..)| ident.cmp(ident2));

    for (name, description, ..) in &lints_rustdoc {
        push_lint_completion(buf, &name.replace('-', "_"), description)
    }
    buf.push_str("];\n\n");

    buf.push_str(r#"pub const RUSTDOC_LINT_GROUPS: &[LintGroup] = &["#);
    for (name, description, children) in &lints_rustdoc {
        if !children.is_empty() {
            push_lint_group(buf, &name.replace('-', "_"), description, children);
        }
    }
    buf.push('\n');
    buf.push_str("];\n");
}

#[track_caller]
fn find_and_slice<'a>(i: &'a str, p: &str) -> &'a str {
    let idx = i.find(p).unwrap();
    &i[idx + p.len()..]
}

/// Parses the unstable book `src_dir` and prints a constant with the list of
/// unstable features into `buf`.
///
/// It does this by looking for all `.md` files in the `language-features` and
/// `library-features` directories, and using the file name as the feature
/// name, and the file contents as the feature description.
fn generate_feature_descriptor(buf: &mut String, src_dir: &Path) {
    let mut features = ["language-features", "library-features"]
        .into_iter()
        .flat_map(|it| list_files(&src_dir.join(it)))
        // Get all `.md` files
        .filter(|path| path.extension() == Some("md".as_ref()))
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

#[allow(clippy::print_stderr)]
fn generate_descriptor_clippy(buf: &mut String, path: &Path) {
    let file_content = std::fs::read_to_string(path).unwrap();
    let mut clippy_lints: Vec<ClippyLint> = Vec::new();
    let mut clippy_groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();

    for line in file_content.lines().map(str::trim) {
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
            let header = "### What it does";
            let line = match line.find(header) {
                Some(idx) => &line[idx + header.len()..],
                None => {
                    let id = &clippy_lints.last().unwrap().id;
                    // these just don't have the common header
                    let allowed = ["allow_attributes", "read_line_without_trim"];
                    if allowed.contains(&id.as_str()) {
                        line
                    } else {
                        eprintln!("\nunexpected clippy prefix for {id}, line={line:?}\n",);
                        continue;
                    }
                }
            };
            // Only take the description, any more than this is a lot of additional data we would embed into the exe
            // which seems unnecessary
            let up_to = line.find(r#"###"#).expect("no second section found?");
            let line = &line[..up_to];

            let clippy_lint = clippy_lints.last_mut().expect("clippy lint must already exist");
            clippy_lint.help = unescape(line).trim().to_owned();
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
            let description = format!("lint group for: {}", children.join(", "));
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

    let children = format!(
        "&[{}]",
        children.iter().map(|it| format!("\"{it}\"")).collect::<Vec<_>>().join(", ")
    );
    format_to!(
        buf,
        r###"
        children: {},
        }},"###,
        children,
    );
}
