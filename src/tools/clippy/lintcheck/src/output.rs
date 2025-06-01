use cargo_metadata::diagnostic::{Diagnostic, DiagnosticSpan};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::Path;
use std::process::ExitStatus;

use crate::config::{LintcheckConfig, OutputFormat};

/// A single emitted output from clippy being executed on a crate. It may either be a
/// `ClippyWarning`, or a `RustcIce` caused by a panic within clippy. A crate may have many
/// `ClippyWarning`s but a maximum of one `RustcIce` (at which point clippy halts execution).
#[derive(Debug)]
pub enum ClippyCheckOutput {
    ClippyWarning(ClippyWarning),
    RustcIce(RustcIce),
}

#[derive(Debug)]
pub struct RustcIce {
    pub crate_name: String,
    pub ice_content: String,
}

impl fmt::Display for RustcIce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:\n{}\n========================================\n",
            self.crate_name, self.ice_content
        )
    }
}

impl RustcIce {
    pub fn from_stderr_and_status(crate_name: &str, status: ExitStatus, stderr: &str) -> Option<Self> {
        if status.code().unwrap_or(0) == 101
        /* ice exit status */
        {
            Some(Self {
                crate_name: crate_name.to_owned(),
                ice_content: stderr.to_owned(),
            })
        } else {
            None
        }
    }
}

/// A single warning that clippy issued while checking a `Crate`
#[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClippyWarning {
    pub name: String,
    pub diag: Diagnostic,
    pub krate: String,
    /// The URL that points to the file and line of the lint emission
    pub url: String,
}

impl ClippyWarning {
    pub fn new(mut diag: Diagnostic, base_url: &str, krate: &str) -> Option<Self> {
        let name = diag.code.clone()?.code;
        if !(name.contains("clippy") || diag.message.contains("clippy"))
            || diag.message.contains("could not read cargo metadata")
        {
            return None;
        }

        // --recursive bypasses cargo so we have to strip the rendered output ourselves
        let rendered = diag.rendered.as_mut().unwrap();
        *rendered = strip_ansi_escapes::strip_str(&rendered);

        // Turns out that there are lints without spans... For example Rust's
        // `renamed_and_removed_lints` if the lint is given via the CLI.
        let span = diag
            .spans
            .iter()
            .find(|span| span.is_primary)
            .or(diag.spans.first())
            .unwrap_or_else(|| panic!("Diagnostic without span: {diag}"));
        let file = &span.file_name;
        let url = if let Some(src_split) = file.find("/src/") {
            // This removes the initial `target/lintcheck/sources/<crate>-<version>/`
            let src_split = src_split + "/src/".len();
            let (_, file) = file.split_at(src_split);

            let line_no = span.line_start;
            base_url.replace("{file}", file).replace("{line}", &line_no.to_string())
        } else {
            file.clone()
        };

        Some(Self {
            name,
            diag,
            krate: krate.to_string(),
            url,
        })
    }

    pub fn span(&self) -> &DiagnosticSpan {
        self.diag.spans.iter().find(|span| span.is_primary).unwrap()
    }

    pub fn to_output(&self, format: OutputFormat) -> String {
        let span = self.span();
        let mut file = span.file_name.clone();
        let file_with_pos = format!("{file}:{}:{}", span.line_start, span.line_end);
        match format {
            OutputFormat::Text => format!("{file_with_pos} {} \"{}\"\n", self.name, self.diag.message),
            OutputFormat::Markdown => {
                if file.starts_with("target") {
                    file.insert_str(0, "../");
                }

                let mut output = String::from("| ");
                write!(output, "[`{file_with_pos}`]({file}#L{})", span.line_start).unwrap();
                write!(output, r#" | `{:<50}` | "{}" |"#, self.name, self.diag.message).unwrap();
                output.push('\n');
                output
            },
            OutputFormat::Json => unreachable!("JSON output is handled via serde"),
        }
    }
}

/// Creates the log file output for [`OutputFormat::Text`] and [`OutputFormat::Markdown`]
pub fn summarize_and_print_changes(
    warnings: &[ClippyWarning],
    ices: &[RustcIce],
    clippy_ver: String,
    config: &LintcheckConfig,
) -> String {
    // generate some stats
    let (stats_formatted, new_stats) = gather_stats(warnings);
    let old_stats = read_stats_from_file(&config.lintcheck_results_path);

    let mut all_msgs: Vec<String> = warnings.iter().map(|warn| warn.to_output(config.format)).collect();
    all_msgs.sort();
    all_msgs.push("\n\n### Stats:\n\n".into());
    all_msgs.push(stats_formatted);

    let mut text = clippy_ver; // clippy version number on top
    text.push_str("\n### Reports\n\n");
    if config.format == OutputFormat::Markdown {
        text.push_str("| file | lint | message |\n");
        text.push_str("| --- | --- | --- |\n");
    }
    write!(text, "{}", all_msgs.join("")).unwrap();
    text.push_str("\n\n### ICEs:\n");
    for ice in ices {
        writeln!(text, "{ice}").unwrap();
    }

    print_stats(old_stats, new_stats, &config.lint_filter);

    text
}

/// Generate a short list of occurring lints-types and their count
fn gather_stats(warnings: &[ClippyWarning]) -> (String, HashMap<&String, usize>) {
    // count lint type occurrences
    let mut counter: HashMap<&String, usize> = HashMap::new();
    for wrn in warnings {
        *counter.entry(&wrn.name).or_insert(0) += 1;
    }

    // collect into a tupled list for sorting
    let mut stats: Vec<(&&String, &usize)> = counter.iter().collect();
    // sort by "000{count} {clippy::lintname}"
    // to not have a lint with 200 and 2 warnings take the same spot
    stats.sort_by_key(|(lint, count)| format!("{count:0>4}, {lint}"));

    let mut header = String::from("| lint                                               | count |\n");
    header.push_str("| -------------------------------------------------- | ----- |\n");
    let stats_string = stats
        .iter()
        .map(|(lint, count)| format!("| {lint:<50} |  {count:>4} |\n"))
        .fold(header, |mut table, line| {
            table.push_str(&line);
            table
        });

    (stats_string, counter)
}

/// read the previous stats from the lintcheck-log file
fn read_stats_from_file(file_path: &Path) -> HashMap<String, usize> {
    let file_content: String = match fs::read_to_string(file_path).ok() {
        Some(content) => content,
        None => {
            return HashMap::new();
        },
    };

    let lines: Vec<String> = file_content.lines().map(ToString::to_string).collect();

    lines
        .iter()
        .skip_while(|line| line.as_str() != "### Stats:")
        // Skipping the table header and the `Stats:` label
        .skip(4)
        .take_while(|line| line.starts_with("| "))
        .filter_map(|line| {
            let mut spl = line.split('|');
            // Skip the first `|` symbol
            spl.next();
            if let (Some(lint), Some(count)) = (spl.next(), spl.next()) {
                Some((lint.trim().to_string(), count.trim().parse::<usize>().unwrap()))
            } else {
                None
            }
        })
        .collect::<HashMap<String, usize>>()
}

/// print how lint counts changed between runs
fn print_stats(old_stats: HashMap<String, usize>, new_stats: HashMap<&String, usize>, lint_filter: &[String]) {
    let same_in_both_hashmaps = old_stats
        .iter()
        .filter(|(old_key, old_val)| new_stats.get::<&String>(old_key) == Some(old_val))
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<Vec<(String, usize)>>();

    let mut old_stats_deduped = old_stats;
    let mut new_stats_deduped = new_stats;

    // remove duplicates from both hashmaps
    for (k, v) in &same_in_both_hashmaps {
        assert!(old_stats_deduped.remove(k) == Some(*v));
        assert!(new_stats_deduped.remove(k) == Some(*v));
    }

    println!("\nStats:");

    // list all new counts  (key is in new stats but not in old stats)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _)| !old_stats_deduped.contains_key::<str>(new_key))
        .for_each(|(new_key, new_value)| {
            println!("{new_key} 0 => {new_value}");
        });

    // list all changed counts (key is in both maps but value differs)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _new_val)| old_stats_deduped.contains_key::<str>(new_key))
        .for_each(|(new_key, new_val)| {
            let old_val = old_stats_deduped.get::<str>(new_key).unwrap();
            println!("{new_key} {old_val} => {new_val}");
        });

    // list all gone counts (key is in old status but not in new stats)
    old_stats_deduped
        .iter()
        .filter(|(old_key, _)| !new_stats_deduped.contains_key::<&String>(old_key))
        .filter(|(old_key, _)| lint_filter.is_empty() || lint_filter.contains(old_key))
        .for_each(|(old_key, old_value)| {
            println!("{old_key} {old_value} => 0");
        });
}
