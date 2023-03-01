use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

mod groups;

pub struct LintExtractor<'a> {
    /// Path to the `src` directory, where it will scan for `.rs` files to
    /// find lint declarations.
    pub src_path: &'a Path,
    /// Path where to save the output.
    pub out_path: &'a Path,
    /// Path to the `rustc` executable.
    pub rustc_path: &'a Path,
    /// The target arch to build the docs for.
    pub rustc_target: &'a str,
    /// Verbose output.
    pub verbose: bool,
    /// Validate the style and the code example.
    pub validate: bool,
}

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
        let blocks: Vec<_> = self.doc.iter().filter(|line| line.starts_with("```rust")).collect();
        !blocks.is_empty() && blocks.iter().all(|line| line.contains(",ignore"))
    }

    /// Checks the doc style of the lint.
    fn check_style(&self) -> Result<(), Box<dyn Error>> {
        for &expected in &["### Example", "### Explanation", "{{produces}}"] {
            if expected == "{{produces}}" && self.is_ignored() {
                if self.doc_contains("{{produces}}") {
                    return Err(format!(
                        "the lint example has `ignore`, but also contains the {{{{produces}}}} marker\n\
                        \n\
                        The documentation generator cannot generate the example output when the \
                        example is ignored.\n\
                        Manually include the sample output below the example. For example:\n\
                        \n\
                        /// ```rust,ignore (needs command line option)\n\
                        /// #[cfg(widnows)]\n\
                        /// fn foo() {{}}\n\
                        /// ```\n\
                        ///\n\
                        /// This will produce:\n\
                        /// \n\
                        /// ```text\n\
                        /// warning: unknown condition name used\n\
                        ///  --> lint_example.rs:1:7\n\
                        ///   |\n\
                        /// 1 | #[cfg(widnows)]\n\
                        ///   |       ^^^^^^^\n\
                        ///   |\n\
                        ///   = note: `#[warn(unexpected_cfgs)]` on by default\n\
                        /// ```\n\
                        \n\
                        Replacing the output with the text of the example you \
                        compiled manually yourself.\n\
                        "
                    ).into());
                }
                continue;
            }
            if !self.doc_contains(expected) {
                return Err(format!("lint docs should contain the line `{}`", expected).into());
            }
        }
        if let Some(first) = self.doc.first() {
            if !first.starts_with(&format!("The `{}` lint", self.name)) {
                return Err(format!(
                    "lint docs should start with the text \"The `{}` lint\" to introduce the lint",
                    self.name
                )
                .into());
            }
        }
        Ok(())
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

impl<'a> LintExtractor<'a> {
    /// Collects all lints, and writes the markdown documentation at the given directory.
    pub fn extract_lint_docs(&self) -> Result<(), Box<dyn Error>> {
        let mut lints = self.gather_lints()?;
        for lint in &mut lints {
            self.generate_output_example(lint).map_err(|e| {
                format!(
                    "failed to test example in lint docs for `{}` in {}:{}: {}",
                    lint.name,
                    lint.path.display(),
                    lint.lineno,
                    e
                )
            })?;
        }
        self.save_lints_markdown(&lints)?;
        self.generate_group_docs(&lints)?;
        Ok(())
    }

    /// Collects all lints from all files in the given directory.
    fn gather_lints(&self) -> Result<Vec<Lint>, Box<dyn Error>> {
        let mut lints = Vec::new();
        for entry in WalkDir::new(self.src_path).into_iter().filter_map(|e| e.ok()) {
            if !entry.path().extension().map_or(false, |ext| ext == "rs") {
                continue;
            }
            lints.extend(self.lints_from_file(entry.path())?);
        }
        if lints.is_empty() {
            return Err("no lints were found!".into());
        }
        Ok(lints)
    }

    /// Collects all lints from the given file.
    fn lints_from_file(&self, path: &Path) -> Result<Vec<Lint>, Box<dyn Error>> {
        let mut lints = Vec::new();
        let contents = fs::read_to_string(path)
            .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
        let mut lines = contents.lines().enumerate();
        'outer: loop {
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
                        if let Some(text) = line.strip_prefix("/// ") {
                            doc_lines.push(text.to_string());
                        } else if line == "///" {
                            doc_lines.push("".to_string());
                        } else if line.starts_with("// ") {
                            // Ignore comments.
                            continue;
                        } else if line.starts_with("#[allow") {
                            // Ignore allow of lints (useful for
                            // invalid_rust_codeblocks).
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
                                if self.validate {
                                    return Err(format!(
                                        "did not find doc lines for lint `{}` in {}",
                                        name,
                                        path.display()
                                    )
                                    .into());
                                } else {
                                    eprintln!(
                                        "warning: lint `{}` in {} does not define any doc lines, \
                                         these are required for the lint documentation",
                                        name,
                                        path.display()
                                    );
                                    continue 'outer;
                                }
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

    /// Mutates the lint definition to replace the `{{produces}}` marker with the
    /// actual output from the compiler.
    fn generate_output_example(&self, lint: &mut Lint) -> Result<(), Box<dyn Error>> {
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
        if self.validate {
            lint.check_style()?;
        }
        // Unfortunately some lints have extra requirements that this simple test
        // setup can't handle (like extern crates). An alternative is to use a
        // separate test suite, and use an include mechanism such as mdbook's
        // `{{#rustdoc_include}}`.
        if !lint.is_ignored() {
            if let Err(e) = self.replace_produces(lint) {
                if self.validate {
                    return Err(e);
                }
                eprintln!(
                    "warning: the code example in lint `{}` in {} failed to \
                     generate the expected output: {}",
                    lint.name,
                    lint.path.display(),
                    e
                );
            }
        }
        Ok(())
    }

    /// Mutates the lint docs to replace the `{{produces}}` marker with the actual
    /// output from the compiler.
    fn replace_produces(&self, lint: &mut Lint) -> Result<(), Box<dyn Error>> {
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
                        let output = self.generate_lint_output(&lint.name, &example, &options)?;
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
        &self,
        name: &str,
        example: &[&mut String],
        options: &[&str],
    ) -> Result<String, Box<dyn Error>> {
        if self.verbose {
            eprintln!("compiling lint {}", name);
        }
        let tempdir = tempfile::TempDir::new()?;
        let tempfile = tempdir.path().join("lint_example.rs");
        let mut source = String::new();
        let needs_main = !example.iter().any(|line| line.contains("fn main"));
        // Remove `# ` prefix for hidden lines.
        let unhidden = example.iter().map(|line| line.strip_prefix("# ").unwrap_or(line));
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
        let mut cmd = Command::new(self.rustc_path);
        if options.contains(&"edition2015") {
            cmd.arg("--edition=2015");
        } else {
            cmd.arg("--edition=2018");
        }
        cmd.arg("--error-format=json");
        cmd.arg("--target").arg(self.rustc_target);
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
        // First try to find the messages with the `code` field set to our lint.
        let matches: Vec<_> = msgs
            .iter()
            .filter(|msg| matches!(&msg["code"]["code"], serde_json::Value::String(s) if s==name))
            .map(|msg| msg["rendered"].as_str().expect("rendered field should exist").to_string())
            .collect();
        if matches.is_empty() {
            // Some lints override their code to something else (E0566).
            // Try to find something that looks like it could be our lint.
            let matches: Vec<_> = msgs.iter().filter(|msg|
                matches!(&msg["rendered"], serde_json::Value::String(s) if s.contains(name)))
                .map(|msg| msg["rendered"].as_str().expect("rendered field should exist").to_string())
                .collect();
            if matches.is_empty() {
                let rendered: Vec<&str> =
                    msgs.iter().filter_map(|msg| msg["rendered"].as_str()).collect();
                let non_json: Vec<&str> =
                    stderr.lines().filter(|line| !line.starts_with('{')).collect();
                Err(format!(
                    "did not find lint `{}` in output of example, got:\n{}\n{}",
                    name,
                    non_json.join("\n"),
                    rendered.join("\n")
                )
                .into())
            } else {
                Ok(matches.join("\n"))
            }
        } else {
            Ok(matches.join("\n"))
        }
    }

    /// Saves the mdbook lint chapters at the given path.
    fn save_lints_markdown(&self, lints: &[Lint]) -> Result<(), Box<dyn Error>> {
        self.save_level(lints, Level::Allow, ALLOWED_MD)?;
        self.save_level(lints, Level::Warn, WARN_MD)?;
        self.save_level(lints, Level::Deny, DENY_MD)?;
        Ok(())
    }

    fn save_level(&self, lints: &[Lint], level: Level, header: &str) -> Result<(), Box<dyn Error>> {
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
        let out_path = self.out_path.join("listing").join(level.doc_filename());
        // Delete the output because rustbuild uses hard links in its copies.
        let _ = fs::remove_file(&out_path);
        fs::write(&out_path, result)
            .map_err(|e| format!("could not write to {}: {}", out_path.display(), e))?;
        Ok(())
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
            if !name.chars().all(|ch| ch.is_uppercase() || ch.is_ascii_digit() || ch == '_')
                || name.is_empty()
            {
                return Err("lint name did not have expected format");
            }
            Ok(name.to_lowercase().to_string())
        }
        None => Err("could not find lint name"),
    }
}

static ALLOWED_MD: &str = r#"# Allowed-by-default Lints

These lints are all set to the 'allow' level by default. As such, they won't show up
unless you set them to a higher lint level with a flag or attribute.

"#;

static WARN_MD: &str = r#"# Warn-by-default Lints

These lints are all set to the 'warn' level by default.

"#;

static DENY_MD: &str = r#"# Deny-by-default Lints

These lints are all set to the 'deny' level by default.

"#;
