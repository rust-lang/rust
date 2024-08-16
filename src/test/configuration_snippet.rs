use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::iter::Enumerate;
use std::path::{Path, PathBuf};

use super::{DIFF_CONTEXT_SIZE, print_mismatches, write_message};
use crate::config::{Config, EmitMode, Verbosity};
use crate::rustfmt_diff::{Mismatch, make_diff};
use crate::{Input, Session};

const CONFIGURATIONS_FILE_NAME: &str = "Configurations.md";

// This enum is used to represent one of three text features in Configurations.md: a block of code
// with its starting line number, the name of a rustfmt configuration option, or the value of a
// rustfmt configuration option.
enum ConfigurationSection {
    CodeBlock((String, u32)), // (String: block of code, u32: line number of code block start)
    ConfigName(String),
    ConfigValue(String),
}

impl ConfigurationSection {
    fn get_section<I: Iterator<Item = String>>(
        file: &mut Enumerate<I>,
    ) -> Option<ConfigurationSection> {
        let config_name_regex = static_regex!(r"^## `([^`]+)`");
        // Configuration values, which will be passed to `from_str`:
        //
        // - must be prefixed with `####`
        // - must be wrapped in backticks
        // - may by wrapped in double quotes (which will be stripped)
        let config_value_regex = static_regex!(r#"^#### `"?([^`]+?)"?`"#);
        loop {
            match file.next() {
                Some((i, line)) => {
                    if line.starts_with("```rust") {
                        // Get the lines of the code block.
                        let lines: Vec<String> = file
                            .map(|(_i, l)| l)
                            .take_while(|l| !l.starts_with("```"))
                            .collect();
                        let block = format!("{}\n", lines.join("\n"));

                        // +1 to translate to one-based indexing
                        // +1 to get to first line of code (line after "```")
                        let start_line = (i + 2) as u32;

                        return Some(ConfigurationSection::CodeBlock((block, start_line)));
                    } else if let Some(c) = config_name_regex.captures(&line) {
                        return Some(ConfigurationSection::ConfigName(String::from(&c[1])));
                    } else if let Some(c) = config_value_regex.captures(&line) {
                        return Some(ConfigurationSection::ConfigValue(String::from(&c[1])));
                    }
                }
                None => return None, // reached the end of the file
            }
        }
    }
}

// This struct stores the information about code blocks in the configurations
// file, formats the code blocks, and prints formatting errors.
struct ConfigCodeBlock {
    config_name: Option<String>,
    config_value: Option<String>,
    code_block: Option<String>,
    code_block_start: Option<u32>,
}

impl ConfigCodeBlock {
    fn new() -> ConfigCodeBlock {
        ConfigCodeBlock {
            config_name: None,
            config_value: None,
            code_block: None,
            code_block_start: None,
        }
    }

    fn set_config_name(&mut self, name: Option<String>) {
        self.config_name = name;
        self.config_value = None;
    }

    fn set_config_value(&mut self, value: Option<String>) {
        self.config_value = value;
    }

    fn set_code_block(&mut self, code_block: String, code_block_start: u32) {
        self.code_block = Some(code_block);
        self.code_block_start = Some(code_block_start);
    }

    fn get_block_config(&self) -> Config {
        let mut config = Config::default();
        config.set().verbose(Verbosity::Quiet);
        if self.config_name.is_some() && self.config_value.is_some() {
            config.override_value(
                self.config_name.as_ref().unwrap(),
                self.config_value.as_ref().unwrap(),
            );
        }
        config
    }

    fn code_block_valid(&self) -> bool {
        // We never expect to not have a code block.
        assert!(self.code_block.is_some() && self.code_block_start.is_some());

        // See if code block begins with #![rustfmt::skip].
        let fmt_skip = self.fmt_skip();

        if self.config_name.is_none() && !fmt_skip {
            write_message(&format!(
                "No configuration name for {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return false;
        }
        if self.config_value.is_none() && !fmt_skip {
            write_message(&format!(
                "No configuration value for {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return false;
        }
        true
    }

    /// True if the code block starts with #![rustfmt::skip]
    fn fmt_skip(&self) -> bool {
        self.code_block
            .as_ref()
            .unwrap()
            .lines()
            .nth(0)
            .unwrap_or("")
            == "#![rustfmt::skip]"
    }

    fn has_parsing_errors<T: Write>(&self, session: &Session<'_, T>) -> bool {
        if session.has_parsing_errors() {
            write_message(&format!(
                "\u{261d}\u{1f3fd} Cannot format {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return true;
        }

        false
    }

    fn print_diff(&self, compare: Vec<Mismatch>) {
        let mut mismatches = HashMap::new();
        mismatches.insert(PathBuf::from(CONFIGURATIONS_FILE_NAME), compare);
        print_mismatches(mismatches, |line_num| {
            format!(
                "\nMismatch at {}:{}:",
                CONFIGURATIONS_FILE_NAME,
                line_num + self.code_block_start.unwrap() - 1
            )
        });
    }

    fn formatted_has_diff(&self, text: &str) -> bool {
        let compare = make_diff(self.code_block.as_ref().unwrap(), text, DIFF_CONTEXT_SIZE);
        if !compare.is_empty() {
            self.print_diff(compare);
            return true;
        }

        false
    }

    // Return a bool indicating if formatting this code block is an idempotent
    // operation. This function also triggers printing any formatting failure
    // messages.
    fn formatted_is_idempotent(&self) -> bool {
        // Verify that we have all of the expected information.
        if !self.code_block_valid() {
            return false;
        }

        let input = Input::Text(self.code_block.as_ref().unwrap().to_owned());
        let mut config = self.get_block_config();
        config.set().emit_mode(EmitMode::Stdout);
        let mut buf: Vec<u8> = vec![];

        {
            let mut session = Session::new(config, Some(&mut buf));
            session.format(input).unwrap();
            if self.has_parsing_errors(&session) {
                return false;
            }
        }

        !self.formatted_has_diff(&String::from_utf8(buf).unwrap())
    }

    // Extract a code block from the iterator. Behavior:
    // - Rust code blocks are identified by lines beginning with "```rust".
    // - One explicit configuration setting is supported per code block.
    // - Rust code blocks with no configuration setting are illegal and cause an
    //   assertion failure, unless the snippet begins with #![rustfmt::skip].
    // - Configuration names in Configurations.md must be in the form of
    //   "## `NAME`".
    // - Configuration values in Configurations.md must be in the form of
    //   "#### `VALUE`".
    fn extract<I: Iterator<Item = String>>(
        file: &mut Enumerate<I>,
        prev: Option<&ConfigCodeBlock>,
        hash_set: &mut HashSet<String>,
    ) -> Option<ConfigCodeBlock> {
        let mut code_block = ConfigCodeBlock::new();
        code_block.config_name = prev.and_then(|cb| cb.config_name.clone());

        loop {
            match ConfigurationSection::get_section(file) {
                Some(ConfigurationSection::CodeBlock((block, start_line))) => {
                    code_block.set_code_block(block, start_line);
                    break;
                }
                Some(ConfigurationSection::ConfigName(name)) => {
                    assert!(
                        Config::is_valid_name(&name),
                        "an unknown configuration option was found: {name}"
                    );
                    assert!(
                        hash_set.remove(&name),
                        "multiple configuration guides found for option {name}"
                    );
                    code_block.set_config_name(Some(name));
                }
                Some(ConfigurationSection::ConfigValue(value)) => {
                    code_block.set_config_value(Some(value));
                }
                None => return None, // end of file was reached
            }
        }

        Some(code_block)
    }
}

#[test]
fn configuration_snippet_tests() {
    super::init_log();
    let blocks = get_code_blocks();
    let failures = blocks
        .iter()
        .filter(|block| !block.fmt_skip())
        .map(ConfigCodeBlock::formatted_is_idempotent)
        .fold(0, |acc, r| acc + (!r as u32));

    // Display results.
    println!("Ran {} configurations tests.", blocks.len());
    assert_eq!(failures, 0, "{failures} configurations tests failed");
}

// Read Configurations.md and build a `Vec` of `ConfigCodeBlock` structs with one
// entry for each Rust code block found.
fn get_code_blocks() -> Vec<ConfigCodeBlock> {
    let mut file_iter = BufReader::new(
        fs::File::open(Path::new(CONFIGURATIONS_FILE_NAME))
            .unwrap_or_else(|_| panic!("couldn't read file {}", CONFIGURATIONS_FILE_NAME)),
    )
    .lines()
    .map(Result::unwrap)
    .enumerate();
    let mut code_blocks: Vec<ConfigCodeBlock> = Vec::new();
    let mut hash_set = Config::hash_set();

    while let Some(cb) = ConfigCodeBlock::extract(&mut file_iter, code_blocks.last(), &mut hash_set)
    {
        code_blocks.push(cb);
    }

    for name in hash_set {
        if !Config::is_hidden_option(&name) {
            panic!("{name} does not have a configuration guide");
        }
    }

    code_blocks
}

#[test]
fn check_unstable_option_tracking_issue_numbers() {
    // Ensure that tracking issue links point to the correct issue number
    let tracking_issue =
        regex::Regex::new(r"\(tracking issue: \[#(?P<number>\d+)\]\((?P<link>\S+)\)\)")
            .expect("failed creating configuration pattern");

    let lines = BufReader::new(
        fs::File::open(Path::new(CONFIGURATIONS_FILE_NAME))
            .unwrap_or_else(|_| panic!("couldn't read file {}", CONFIGURATIONS_FILE_NAME)),
    )
    .lines()
    .map(Result::unwrap)
    .enumerate();

    for (idx, line) in lines {
        if let Some(capture) = tracking_issue.captures(&line) {
            let number = capture.name("number").unwrap().as_str();
            let link = capture.name("link").unwrap().as_str();
            assert!(
                link.ends_with(number),
                "{} on line {} does not point to issue #{}",
                link,
                idx + 1,
                number,
            );
        }
    }
}
