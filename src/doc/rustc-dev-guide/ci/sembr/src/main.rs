use std::path::PathBuf;
use std::sync::LazyLock;
use std::{fs, process};

use anyhow::Result;
use clap::Parser;
use ignore::Walk;
use imara_diff::{Algorithm, BasicLineDiffPrinter, Diff, InternedInput, UnifiedDiffConfig};
use regex::Regex;

#[derive(Parser)]
struct Cli {
    /// File or directory to check
    path: PathBuf,
    #[arg(long)]
    /// Modify files that do not comply
    overwrite: bool,
    /// Applies to lines that are to be split
    #[arg(long, default_value_t = 100)]
    line_length_limit: usize,
    #[arg(long)]
    show_diff: bool,
}

static REGEX_IGNORE_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\.|\?|;|!|,|\-)$").unwrap());
static REGEX_IGNORE_LINK_TARGETS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[.+\]: ").unwrap());
static REGEX_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([^\.\d\-\*]\.|[^r]\?|!)\s").unwrap());
// list elements, numbered (1.) or not  (- and *)
static REGEX_LIST_ENTRY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(\d\.|\-|\*)\s+").unwrap());

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut compliant = Vec::new();
    let mut not_compliant = Vec::new();
    let mut made_compliant = Vec::new();
    for result in Walk::new(cli.path) {
        let entry = result?;
        if entry.file_type().expect("no stdin").is_dir() {
            continue;
        }
        let path = entry.into_path();
        if let Some(extension) = path.extension() {
            if extension != "md" {
                continue;
            }
            let old = fs::read_to_string(&path)?;
            let new = lengthen_lines(&comply(&old), cli.line_length_limit);
            if new == old {
                compliant.push(path.clone());
            } else if cli.overwrite {
                fs::write(&path, new)?;
                made_compliant.push(path.clone());
            } else if cli.show_diff {
                println!("{}:", path.display());
                show_diff(&old, &new);
                println!("---");
            } else {
                not_compliant.push(path.clone());
            }
        }
    }
    if !compliant.is_empty() {
        display("compliant", &compliant);
    }
    if !made_compliant.is_empty() {
        display("made compliant", &made_compliant);
    }
    if !not_compliant.is_empty() {
        display("not compliant", &not_compliant);
        process::exit(1);
    }
    Ok(())
}

fn show_diff(old: &str, new: &str) {
    let input = InternedInput::new(old, new);
    let mut diff = Diff::compute(Algorithm::Histogram, &input);
    diff.postprocess_lines(&input);
    let diff = diff
        .unified_diff(&BasicLineDiffPrinter(&input.interner), UnifiedDiffConfig::default(), &input)
        .to_string();
    print!("{diff}");
}

fn display(header: &str, paths: &[PathBuf]) {
    println!("{header}:");
    for element in paths {
        println!("- {}", element.display());
    }
}

fn ignore(line: &str, in_code_block: bool) -> bool {
    in_code_block
        || line.to_lowercase().contains("e.g.")
        || line.contains("i.e.")
        || line.contains('|')
        || line.trim_start().starts_with('>')
        || line.starts_with('#')
        || line.trim().is_empty()
        || REGEX_IGNORE_LINK_TARGETS.is_match(line)
}

fn comply(content: &str) -> String {
    let content: Vec<_> = content.lines().map(std::borrow::ToOwned::to_owned).collect();
    let mut new_content = content.clone();
    let mut new_n = 0;
    let mut in_code_block = false;
    for (n, line) in content.into_iter().enumerate() {
        if n != 0 {
            new_n += 1;
        }
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if ignore(&line, in_code_block) {
            continue;
        }
        if REGEX_SPLIT.is_match(&line) {
            let indent = if let Some(regex_match) = REGEX_LIST_ENTRY.find(&line) {
                regex_match.len()
            } else {
                line.find(|ch: char| !ch.is_whitespace()).unwrap()
            };
            let mut newly_split_lines = line.split_inclusive(&*REGEX_SPLIT);
            let first = newly_split_lines.next().unwrap().trim_end().to_owned();
            let mut remaining: Vec<_> = newly_split_lines
                .map(|portion| format!("{:indent$}{}", "", portion.trim_end()))
                .collect();
            let mut new_lines = Vec::new();
            new_lines.push(first);
            new_lines.append(&mut remaining);
            new_content.splice(new_n..=new_n, new_lines.clone());
            new_n += new_lines.len() - 1;
        }
    }
    new_content.join("\n") + "\n"
}

fn lengthen_lines(content: &str, limit: usize) -> String {
    let content: Vec<_> = content.lines().map(std::borrow::ToOwned::to_owned).collect();
    let mut new_content = content.clone();
    let mut new_n = 0;
    let mut in_code_block = false;
    let mut in_html_div = false;
    let mut skip_next = false;
    for (n, line) in content.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if n != 0 {
            new_n += 1;
        }
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if line.trim_start().starts_with("<div") {
            in_html_div = true;
            continue;
        }
        if line.trim_start().starts_with("</div") {
            in_html_div = false;
            continue;
        }
        if in_html_div {
            continue;
        }
        if ignore(line, in_code_block) || REGEX_SPLIT.is_match(line) {
            continue;
        }
        let Some(next_line) = content.get(n + 1) else {
            continue;
        };
        if ignore(next_line, in_code_block)
            || REGEX_LIST_ENTRY.is_match(next_line)
            || REGEX_IGNORE_END.is_match(line)
        {
            continue;
        }
        if line.len() + next_line.len() < limit {
            new_content[new_n] = format!("{line} {}", next_line.trim_start());
            new_content.remove(new_n + 1);
            skip_next = true;
        }
    }
    new_content.join("\n") + "\n"
}

#[test]
fn test_sembr() {
    let original = "
# some. heading
must! be. split?
1. ignore a dot after number. but no further
ignore | tables
ignore e.g. and
ignore i.e. and
ignore E.g. too
- list. entry
 * list. entry
```
some code. block
```
sentence with *italics* should not be ignored. truly.
git log main.. compiler
 foo.   bar.  baz
";
    let expected = "
# some. heading
must!
be.
split?
1. ignore a dot after number.
   but no further
ignore | tables
ignore e.g. and
ignore i.e. and
ignore E.g. too
- list.
  entry
 * list.
   entry
```
some code. block
```
sentence with *italics* should not be ignored.
truly.
git log main.. compiler
 foo.
   bar.
  baz
";
    assert_eq!(expected, comply(original));
}

#[test]
fn test_prettify() {
    let original = "\
do not split
short sentences
<div class='warning'>
a bit of text inside
</div>
preserve next line
1. one

preserve next line
- two

preserve next line
* three
";
    let expected = "\
do not split short sentences
<div class='warning'>
a bit of text inside
</div>
preserve next line
1. one

preserve next line
- two

preserve next line
* three
";
    assert_eq!(expected, lengthen_lines(original, 50));
}

#[test]
fn test_prettify_prefix_spaces() {
    let original = "\
 do not split
 short sentences
";
    let expected = "\
 do not split short sentences
";
    assert_eq!(expected, lengthen_lines(original, 50));
}

#[test]
fn test_prettify_ignore_link_targets() {
    let original = "\
[a target]: https://example.com
[another target]: https://example.com
";
    assert_eq!(original, lengthen_lines(original, 100));
}

#[test]
fn test_sembr_then_prettify() {
    let original = "
hi there. do
not split
short sentences.
hi again.
";
    let expected = "
hi there.
do
not split
short sentences.
hi again.
";
    let processed = comply(original);
    assert_eq!(expected, processed);
    let expected = "
hi there.
do not split
short sentences.
hi again.
";
    let processed = lengthen_lines(&processed, 50);
    assert_eq!(expected, processed);
    let expected = "
hi there.
do not split short sentences.
hi again.
";
    let processed = lengthen_lines(&processed, 50);
    assert_eq!(expected, processed);
}

#[test]
fn test_sembr_question_mark() {
    let original = "
o? whatever
r? @reviewer
 r? @reviewer
";
    let expected = "
o?
whatever
r? @reviewer
 r? @reviewer
";
    assert_eq!(expected, comply(original));
}
