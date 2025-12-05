//! Generates descriptor structures for unstable features from the unstable book
//! and lints from rustc, rustdoc, and clippy.
#![allow(clippy::disallowed_types)]

use std::{
    collections::{HashMap, hash_map},
    fs,
    path::Path,
    str::FromStr,
};

use edition::Edition;
use stdx::format_to;
use xshell::{Shell, cmd};

use crate::{
    codegen::{add_preamble, ensure_file_contents, reformat},
    project_root,
    util::list_files,
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
         compiler library src/tools src/doc/book"
    )
    .run()
    .unwrap();

    let mut contents = String::from(
        r"
use span::Edition;

use crate::Severity;

#[derive(Clone)]
pub struct Lint {
    pub label: &'static str,
    pub description: &'static str,
    pub default_severity: Severity,
    pub warn_since: Option<Edition>,
    pub deny_since: Option<Edition>,
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
        "curl https://rust-lang.github.io/rust-clippy/stable/lints.json --output {lints_json}"
    )
    .run()
    .unwrap();
    generate_descriptor_clippy(&mut contents, &lints_json);

    let contents = add_preamble(crate::flags::CodegenType::LintDefinitions, reformat(contents));

    let destination = project_root().join(DESTINATION);
    ensure_file_contents(
        crate::flags::CodegenType::LintDefinitions,
        destination.as_path(),
        &contents,
        check,
    );
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Allow,
    Warn,
    Deny,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Severity::{}",
            match self {
                Severity::Allow => "Allow",
                Severity::Warn => "Warning",
                Severity::Deny => "Error",
            }
        )
    }
}

impl FromStr for Severity {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "allow" => Ok(Self::Allow),
            "warn" => Ok(Self::Warn),
            "deny" => Ok(Self::Deny),
            _ => Err("invalid severity"),
        }
    }
}

#[derive(Debug)]
struct Lint {
    description: String,
    default_severity: Severity,
    warn_since: Option<Edition>,
    deny_since: Option<Edition>,
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
    fn get_lints_as_text(
        stdout: &str,
    ) -> (
        impl Iterator<Item = (String, &str, Severity)> + '_,
        impl Iterator<Item = (String, Lint, impl Iterator<Item = String> + '_)> + '_,
        impl Iterator<Item = (String, &str, Severity)> + '_,
        impl Iterator<Item = (String, Lint, impl Iterator<Item = String> + '_)> + '_,
    ) {
        let lints_pat = "----  -------  -------\n";
        let lint_groups_pat = "----  ---------\n";
        let lints = find_and_slice(stdout, lints_pat);
        let lint_groups = find_and_slice(lints, lint_groups_pat);
        let lints_rustdoc = find_and_slice(lint_groups, lints_pat);
        let lint_groups_rustdoc = find_and_slice(lints_rustdoc, lint_groups_pat);

        let lints = lints.lines().take_while(|l| !l.is_empty()).map(|line| {
            let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
            let (severity, description) = rest.trim().split_once(char::is_whitespace).unwrap();
            (name.trim().replace('-', "_"), description.trim(), severity.parse().unwrap())
        });
        let lint_groups = lint_groups.lines().take_while(|l| !l.is_empty()).map(|line| {
            let (name, lints) = line.trim().split_once(char::is_whitespace).unwrap();
            let label = name.trim().replace('-', "_");
            let lint = Lint {
                description: format!("lint group for: {}", lints.trim()),
                default_severity: Severity::Allow,
                warn_since: None,
                deny_since: None,
            };
            let children = lints
                .split_ascii_whitespace()
                .map(|s| s.trim().trim_matches(',').replace('-', "_"));
            (label, lint, children)
        });

        let lints_rustdoc = lints_rustdoc.lines().take_while(|l| !l.is_empty()).map(|line| {
            let (name, rest) = line.trim().split_once(char::is_whitespace).unwrap();
            let (severity, description) = rest.trim().split_once(char::is_whitespace).unwrap();
            (name.trim().replace('-', "_"), description.trim(), severity.parse().unwrap())
        });
        let lint_groups_rustdoc =
            lint_groups_rustdoc.lines().take_while(|l| !l.is_empty()).map(|line| {
                let (name, lints) = line.trim().split_once(char::is_whitespace).unwrap();
                let label = name.trim().replace('-', "_");
                let lint = Lint {
                    description: format!("lint group for: {}", lints.trim()),
                    default_severity: Severity::Allow,
                    warn_since: None,
                    deny_since: None,
                };
                let children = lints
                    .split_ascii_whitespace()
                    .map(|s| s.trim().trim_matches(',').replace('-', "_"));
                (label, lint, children)
            });

        (lints, lint_groups, lints_rustdoc, lint_groups_rustdoc)
    }

    fn insert_lints<'a>(
        edition: Edition,
        lints_map: &mut HashMap<String, Lint>,
        lint_groups_map: &mut HashMap<String, (Lint, Vec<String>)>,
        lints: impl Iterator<Item = (String, &'a str, Severity)>,
        lint_groups: impl Iterator<Item = (String, Lint, impl Iterator<Item = String>)>,
    ) {
        for (lint_name, lint_description, lint_severity) in lints {
            let lint = lints_map.entry(lint_name).or_insert_with(|| Lint {
                description: lint_description.to_owned(),
                default_severity: Severity::Allow,
                warn_since: None,
                deny_since: None,
            });
            if lint_severity == Severity::Warn
                && lint.warn_since.is_none()
                && lint.default_severity < Severity::Warn
            {
                lint.warn_since = Some(edition);
            }
            if lint_severity == Severity::Deny
                && lint.deny_since.is_none()
                && lint.default_severity < Severity::Deny
            {
                lint.deny_since = Some(edition);
            }
        }

        for (group_name, lint, children) in lint_groups {
            match lint_groups_map.entry(group_name) {
                hash_map::Entry::Vacant(entry) => {
                    entry.insert((lint, Vec::from_iter(children)));
                }
                hash_map::Entry::Occupied(mut entry) => {
                    // Overwrite, because some groups (such as edition incompatibility) are changed.
                    *entry.get_mut() = (lint, Vec::from_iter(children));
                }
            }
        }
    }

    fn get_lints(
        sh: &Shell,
        edition: Edition,
        lints_map: &mut HashMap<String, Lint>,
        lint_groups_map: &mut HashMap<String, (Lint, Vec<String>)>,
        lints_rustdoc_map: &mut HashMap<String, Lint>,
        lint_groups_rustdoc_map: &mut HashMap<String, (Lint, Vec<String>)>,
    ) {
        let edition_str = edition.to_string();
        let stdout = cmd!(sh, "rustdoc +nightly -Whelp -Zunstable-options --edition={edition_str}")
            .read()
            .unwrap();
        let (lints, lint_groups, lints_rustdoc, lint_groups_rustdoc) = get_lints_as_text(&stdout);

        insert_lints(edition, lints_map, lint_groups_map, lints, lint_groups);
        insert_lints(
            edition,
            lints_rustdoc_map,
            lint_groups_rustdoc_map,
            lints_rustdoc,
            lint_groups_rustdoc,
        );
    }

    let basic_lints = cmd!(sh, "rustdoc +nightly -Whelp --edition=2015").read().unwrap();
    let (lints, lint_groups, lints_rustdoc, lint_groups_rustdoc) = get_lints_as_text(&basic_lints);

    let mut lints = lints
        .map(|(label, description, severity)| {
            (
                label,
                Lint {
                    description: description.to_owned(),
                    default_severity: severity,
                    warn_since: None,
                    deny_since: None,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let mut lint_groups = lint_groups
        .map(|(label, lint, children)| (label, (lint, Vec::from_iter(children))))
        .collect::<HashMap<_, _>>();
    let mut lints_rustdoc = lints_rustdoc
        .map(|(label, description, severity)| {
            (
                label,
                Lint {
                    description: description.to_owned(),
                    default_severity: severity,
                    warn_since: None,
                    deny_since: None,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let mut lint_groups_rustdoc = lint_groups_rustdoc
        .map(|(label, lint, children)| (label, (lint, Vec::from_iter(children))))
        .collect::<HashMap<_, _>>();

    for edition in Edition::iter().skip(1) {
        get_lints(
            sh,
            edition,
            &mut lints,
            &mut lint_groups,
            &mut lints_rustdoc,
            &mut lint_groups_rustdoc,
        );
    }

    let mut lints = Vec::from_iter(lints);
    lints.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut lint_groups = Vec::from_iter(lint_groups);
    lint_groups.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut lints_rustdoc = Vec::from_iter(lints_rustdoc);
    lints_rustdoc.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut lint_groups_rustdoc = Vec::from_iter(lint_groups_rustdoc);
    lint_groups_rustdoc.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    buf.push_str(r#"pub const DEFAULT_LINTS: &[Lint] = &["#);
    buf.push('\n');

    for (name, lint) in &lints {
        push_lint_completion(buf, name, lint);
    }
    for (name, (group, _)) in &lint_groups {
        push_lint_completion(buf, name, group);
    }
    buf.push_str("];\n\n");

    buf.push_str(r#"pub const DEFAULT_LINT_GROUPS: &[LintGroup] = &["#);
    for (name, (lint, children)) in &lint_groups {
        if name == "warnings" {
            continue;
        }
        push_lint_group(buf, name, lint, children);
    }
    buf.push('\n');
    buf.push_str("];\n");

    // rustdoc

    buf.push('\n');
    buf.push_str(r#"pub const RUSTDOC_LINTS: &[Lint] = &["#);
    buf.push('\n');

    for (name, lint) in &lints_rustdoc {
        push_lint_completion(buf, name, lint);
    }
    for (name, (group, _)) in &lint_groups_rustdoc {
        push_lint_completion(buf, name, group);
    }
    buf.push_str("];\n\n");

    buf.push_str(r#"pub const RUSTDOC_LINT_GROUPS: &[LintGroup] = &["#);
    for (name, (lint, children)) in &lint_groups_rustdoc {
        push_lint_group(buf, name, lint, children);
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
        let lint = Lint {
            description: doc,
            default_severity: Severity::Allow,
            warn_since: None,
            deny_since: None,
        };
        push_lint_completion(buf, &feature_ident, &lint);
    }
    buf.push('\n');
    buf.push_str("];\n");
}

#[derive(Debug, Default)]
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
            unescape(line).trim().clone_into(&mut clippy_lint.help);
        }
    }
    clippy_lints.sort_by(|lint, lint2| lint.id.cmp(&lint2.id));

    buf.push_str(r#"pub const CLIPPY_LINTS: &[Lint] = &["#);
    buf.push('\n');
    for clippy_lint in clippy_lints.into_iter() {
        let lint_ident = format!("clippy::{}", clippy_lint.id);
        let lint = Lint {
            description: clippy_lint.help,
            // Allow clippy lints by default, not all users want them.
            default_severity: Severity::Allow,
            warn_since: None,
            deny_since: None,
        };
        push_lint_completion(buf, &lint_ident, &lint);
    }
    buf.push_str("];\n");

    buf.push_str(r#"pub const CLIPPY_LINT_GROUPS: &[LintGroup] = &["#);
    for (id, children) in clippy_groups {
        let children = children.iter().map(|id| format!("clippy::{id}")).collect::<Vec<_>>();
        if !children.is_empty() {
            let lint_ident = format!("clippy::{id}");
            let description = format!("lint group for: {}", children.join(", "));
            let lint = Lint {
                description,
                default_severity: Severity::Allow,
                warn_since: None,
                deny_since: None,
            };
            push_lint_group(buf, &lint_ident, &lint, &children);
        }
    }
    buf.push('\n');
    buf.push_str("];\n");
}

fn push_lint_completion(buf: &mut String, name: &str, lint: &Lint) {
    format_to!(
        buf,
        r###"    Lint {{
        label: "{}",
        description: r##"{}"##,
        default_severity: {},
        warn_since: "###,
        name,
        lint.description,
        lint.default_severity,
    );
    match lint.warn_since {
        Some(edition) => format_to!(buf, "Some(Edition::Edition{edition})"),
        None => buf.push_str("None"),
    }
    format_to!(
        buf,
        r###",
        deny_since: "###
    );
    match lint.deny_since {
        Some(edition) => format_to!(buf, "Some(Edition::Edition{edition})"),
        None => buf.push_str("None"),
    }
    format_to!(
        buf,
        r###",
    }},"###
    );
}

fn push_lint_group(buf: &mut String, name: &str, lint: &Lint, children: &[String]) {
    buf.push_str(
        r###"    LintGroup {
        lint:
        "###,
    );

    push_lint_completion(buf, name, lint);

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
