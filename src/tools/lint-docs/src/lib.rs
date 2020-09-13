use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

mod groups;

struct Lint {
    name: String,
    doc: Vec<String>,
    level: Level,
    path: PathBuf,
    lineno: usize,
}

impl Lint {
    fn doc_contains(&self, text: &str) -> bool {
        self.doc.iter().any(|line| line.contains(text))
    }

    fn is_ignored(&self) -> bool {
        self.doc
            .iter()
            .filter(|line| line.starts_with("```rust"))
            .all(|line| line.contains(",ignore"))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Allow,
    Warn,
    Deny,
}

impl Level {
    fn doc_filename(&self) -> &str {
        match self {
            Level::Allow => "allowed-by-default.md",
            Level::Warn => "warn-by-default.md",
            Level::Deny => "deny-by-default.md",
        }
    }
}

/// Collects all lints, and writes the markdown documentation at the given directory.
pub fn extract_lint_docs(
    src_path: &Path,
    out_path: &Path,
    rustc_path: &Path,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut lints = gather_lints(src_path)?;
    for lint in &mut lints {
        generate_output_example(lint, rustc_path, verbose).map_err(|e| {
            format!(
                "failed to test example in lint docs for `{}` in {}:{}: {}",
                lint.name,
                lint.path.display(),
                lint.lineno,
                e
            )
        })?;
    }
    save_lints_markdown(&lints, &out_path.join("listing"))?;
    groups::generate_group_docs(&lints, rustc_path, out_path)?;
    Ok(())
}

/// Collects all lints from all files in the given directory.
fn gather_lints(src_path: &Path) -> Result<Vec<Lint>, Box<dyn Error>> {
    let mut lints = Vec::new();
    for entry in WalkDir::new(src_path).into_iter().filter_map(|e| e.ok()) {
        if !entry.path().extension().map_or(false, |ext| ext == "rs") {
            continue;
        }
        lints.extend(lints_from_file(entry.path())?);
    }
    if lints.is_empty() {
        return Err("no lints were found!".into());
    }
    Ok(lints)
}

/// Collects all lints from the given file.
fn lints_from_file(path: &Path) -> Result<Vec<Lint>, Box<dyn Error>> {
    let mut lints = Vec::new();
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
    let mut lines = contents.lines().enumerate();
    loop {
        // Find a lint declaration.
        let lint_start = loop {
            match lines.next() {
                Some((lineno, line)) => {
                    if line.trim().starts_with("declare_lint!") {
                        break lineno + 1;
                    }
                }
                None => return Ok(lints),
            }
        };
        // Read the lint.
        let mut doc_lines = Vec::new();
        let (doc, name) = loop {
            match lines.next() {
                Some((lineno, line)) => {
                    let line = line.trim();
                    if line.starts_with("/// ") {
                        doc_lines.push(line.trim()[4..].to_string());
                    } else if line.starts_with("///") {
                        doc_lines.push("".to_string());
                    } else if line.starts_with("// ") {
                        // Ignore comments.
                        continue;
                    } else {
                        let name = lint_name(line).map_err(|e| {
                            format!(
                                "could not determine lint name in {}:{}: {}, line was `{}`",
                                path.display(),
                                lineno,
                                e,
                                line
                            )
                        })?;
                        if doc_lines.is_empty() {
                            return Err(format!(
                                "did not find doc lines for lint `{}` in {}",
                                name,
                                path.display()
                            )
                            .into());
                        }
                        break (doc_lines, name);
                    }
                }
                None => {
                    return Err(format!(
                        "unexpected EOF for lint definition at {}:{}",
                        path.display(),
                        lint_start
                    )
                    .into());
                }
            }
        };
        // These lints are specifically undocumented. This should be reserved
        // for internal rustc-lints only.
        if name == "deprecated_in_future" {
            continue;
        }
        // Read the level.
        let level = loop {
            match lines.next() {
                // Ignore comments.
                Some((_, line)) if line.trim().starts_with("// ") => {}
                Some((lineno, line)) => match line.trim() {
                    "Allow," => break Level::Allow,
                    "Warn," => break Level::Warn,
                    "Deny," => break Level::Deny,
                    _ => {
                        return Err(format!(
                            "unexpected lint level `{}` in {}:{}",
                            line,
                            path.display(),
                            lineno
                        )
                        .into());
                    }
                },
                None => {
                    return Err(format!(
                        "expected lint level in {}:{}, got EOF",
                        path.display(),
                        lint_start
                    )
                    .into());
                }
            }
        };
        // The rest of the lint definition is ignored.
        assert!(!doc.is_empty());
        lints.push(Lint { name, doc, level, path: PathBuf::from(path), lineno: lint_start });
    }
}

/// Extracts the lint name (removing the visibility modifier, and checking validity).
fn lint_name(line: &str) -> Result<String, &'static str> {
    // Skip over any potential `pub` visibility.
    match line.trim().split(' ').next_back() {
        Some(name) => {
            if !name.ends_with(',') {
                return Err("lint name should end with comma");
            }
            let name = &name[..name.len() - 1];
            if !name.chars().all(|ch| ch.is_uppercase() || ch == '_') || name.is_empty() {
                return Err("lint name did not have expected format");
            }
            Ok(name.to_lowercase().to_string())
        }
        None => Err("could not find lint name"),
    }
}

/// Mutates the lint definition to replace the `{{produces}}` marker with the
/// actual output from the compiler.
fn generate_output_example(
    lint: &mut Lint,
    rustc_path: &Path,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    // Explicit list of lints that are allowed to not have an example. Please
    // try to avoid adding to this list.
    if matches!(
        lint.name.as_str(),
        "unused_features" // broken lint
        | "unstable_features" // deprecated
    ) {
        return Ok(());
    }
    if lint.doc_contains("[rustdoc book]") && !lint.doc_contains("{{produces}}") {
        // Rustdoc lints are documented in the rustdoc book, don't check these.
        return Ok(());
    }
    check_style(lint)?;
    // Unfortunately some lints have extra requirements that this simple test
    // setup can't handle (like extern crates). An alternative is to use a
    // separate test suite, and use an include mechanism such as mdbook's
    // `{{#rustdoc_include}}`.
    if !lint.is_ignored() {
        replace_produces(lint, rustc_path, verbose)?;
    }
    Ok(())
}

/// Checks the doc style of the lint.
fn check_style(lint: &Lint) -> Result<(), Box<dyn Error>> {
    for &expected in &["### Example", "### Explanation", "{{produces}}"] {
        if expected == "{{produces}}" && lint.is_ignored() {
            continue;
        }
        if !lint.doc_contains(expected) {
            return Err(format!("lint docs should contain the line `{}`", expected).into());
        }
    }
    if let Some(first) = lint.doc.first() {
        if !first.starts_with(&format!("The `{}` lint", lint.name)) {
            return Err(format!(
                "lint docs should start with the text \"The `{}` lint\" to introduce the lint",
                lint.name
            )
            .into());
        }
    }
    Ok(())
}

/// Mutates the lint docs to replace the `{{produces}}` marker with the actual
/// output from the compiler.
fn replace_produces(
    lint: &mut Lint,
    rustc_path: &Path,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut lines = lint.doc.iter_mut();
    loop {
        // Find start of example.
        let options = loop {
            match lines.next() {
                Some(line) if line.starts_with("```rust") => {
                    break line[7..].split(',').collect::<Vec<_>>();
                }
                Some(line) if line.contains("{{produces}}") => {
                    return Err("lint marker {{{{produces}}}} found, \
                        but expected to immediately follow a rust code block"
                        .into());
                }
                Some(_) => {}
                None => return Ok(()),
            }
        };
        // Find the end of example.
        let mut example = Vec::new();
        loop {
            match lines.next() {
                Some(line) if line == "```" => break,
                Some(line) => example.push(line),
                None => {
                    return Err(format!(
                        "did not find end of example triple ticks ```, docs were:\n{:?}",
                        lint.doc
                    )
                    .into());
                }
            }
        }
        // Find the {{produces}} line.
        loop {
            match lines.next() {
                Some(line) if line.is_empty() => {}
                Some(line) if line == "{{produces}}" => {
                    let output =
                        generate_lint_output(&lint.name, &example, &options, rustc_path, verbose)?;
                    line.replace_range(
                        ..,
                        &format!(
                            "This will produce:\n\
                        \n\
                        ```text\n\
                        {}\
                        ```",
                            output
                        ),
                    );
                    break;
                }
                // No {{produces}} after example, find next example.
                Some(_line) => break,
                None => return Ok(()),
            }
        }
    }
}

/// Runs the compiler against the example, and extracts the output.
fn generate_lint_output(
    name: &str,
    example: &[&mut String],
    options: &[&str],
    rustc_path: &Path,
    verbose: bool,
) -> Result<String, Box<dyn Error>> {
    if verbose {
        eprintln!("compiling lint {}", name);
    }
    let tempdir = tempfile::TempDir::new()?;
    let tempfile = tempdir.path().join("lint_example.rs");
    let mut source = String::new();
    let needs_main = !example.iter().any(|line| line.contains("fn main"));
    // Remove `# ` prefix for hidden lines.
    let unhidden =
        example.iter().map(|line| if line.starts_with("# ") { &line[2..] } else { line });
    let mut lines = unhidden.peekable();
    while let Some(line) = lines.peek() {
        if line.starts_with("#!") {
            source.push_str(line);
            source.push('\n');
            lines.next();
        } else {
            break;
        }
    }
    if needs_main {
        source.push_str("fn main() {\n");
    }
    for line in lines {
        source.push_str(line);
        source.push('\n')
    }
    if needs_main {
        source.push_str("}\n");
    }
    fs::write(&tempfile, source)
        .map_err(|e| format!("failed to write {}: {}", tempfile.display(), e))?;
    let mut cmd = Command::new(rustc_path);
    if options.contains(&"edition2015") {
        cmd.arg("--edition=2015");
    } else {
        cmd.arg("--edition=2018");
    }
    cmd.arg("--error-format=json");
    if options.contains(&"test") {
        cmd.arg("--test");
    }
    cmd.arg("lint_example.rs");
    cmd.current_dir(tempdir.path());
    let output = cmd.output().map_err(|e| format!("failed to run command {:?}\n{}", cmd, e))?;
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    let msgs = stderr
        .lines()
        .filter(|line| line.starts_with('{'))
        .map(serde_json::from_str)
        .collect::<Result<Vec<serde_json::Value>, _>>()?;
    match msgs
        .iter()
        .find(|msg| matches!(&msg["code"]["code"], serde_json::Value::String(s) if s==name))
    {
        Some(msg) => {
            let rendered = msg["rendered"].as_str().expect("rendered field should exist");
            Ok(rendered.to_string())
        }
        None => {
            match msgs.iter().find(
                |msg| matches!(&msg["rendered"], serde_json::Value::String(s) if s.contains(name)),
            ) {
                Some(msg) => {
                    let rendered = msg["rendered"].as_str().expect("rendered field should exist");
                    Ok(rendered.to_string())
                }
                None => {
                    let rendered: Vec<&str> =
                        msgs.iter().filter_map(|msg| msg["rendered"].as_str()).collect();
                    Err(format!(
                        "did not find lint `{}` in output of example, got:\n{}",
                        name,
                        rendered.join("\n")
                    )
                    .into())
                }
            }
        }
    }
}

static ALLOWED_MD: &str = r#"# Allowed-by-default lints

These lints are all set to the 'allow' level by default. As such, they won't show up
unless you set them to a higher lint level with a flag or attribute.

"#;

static WARN_MD: &str = r#"# Warn-by-default lints

These lints are all set to the 'warn' level by default.

"#;

static DENY_MD: &str = r#"# Deny-by-default lints

These lints are all set to the 'deny' level by default.

"#;

/// Saves the mdbook lint chapters at the given path.
fn save_lints_markdown(lints: &[Lint], out_dir: &Path) -> Result<(), Box<dyn Error>> {
    save_level(lints, Level::Allow, out_dir, ALLOWED_MD)?;
    save_level(lints, Level::Warn, out_dir, WARN_MD)?;
    save_level(lints, Level::Deny, out_dir, DENY_MD)?;
    Ok(())
}

fn save_level(
    lints: &[Lint],
    level: Level,
    out_dir: &Path,
    header: &str,
) -> Result<(), Box<dyn Error>> {
    let mut result = String::new();
    result.push_str(header);
    let mut these_lints: Vec<_> = lints.iter().filter(|lint| lint.level == level).collect();
    these_lints.sort_unstable_by_key(|lint| &lint.name);
    for lint in &these_lints {
        write!(result, "* [`{}`](#{})\n", lint.name, lint.name.replace("_", "-")).unwrap();
    }
    result.push('\n');
    for lint in &these_lints {
        write!(result, "## {}\n\n", lint.name.replace("_", "-")).unwrap();
        for line in &lint.doc {
            result.push_str(line);
            result.push('\n');
        }
        result.push('\n');
    }
    let out_path = out_dir.join(level.doc_filename());
    // Delete the output because rustbuild uses hard links in its copies.
    let _ = fs::remove_file(&out_path);
    fs::write(&out_path, result)
        .map_err(|e| format!("could not write to {}: {}", out_path.display(), e))?;
    Ok(())
}
