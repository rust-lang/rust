// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::codemap::FileName;

use config::config_type::ConfigType;
use config::file_lines::FileLines;
use config::lists::*;
use config::Config;
use {FmtResult, WRITE_MODE_LIST};

use failure::err_msg;

use getopts::Matches;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Macro for deriving implementations of Serialize/Deserialize for enums
#[macro_export]
macro_rules! impl_enum_serialize_and_deserialize {
    ( $e:ident, $( $x:ident ),* ) => {
        impl ::serde::ser::Serialize for $e {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: ::serde::ser::Serializer
            {
                use serde::ser::Error;

                // We don't know whether the user of the macro has given us all options.
                #[allow(unreachable_patterns)]
                match *self {
                    $(
                        $e::$x => serializer.serialize_str(stringify!($x)),
                    )*
                    _ => {
                        Err(S::Error::custom(format!("Cannot serialize {:?}", self)))
                    }
                }
            }
        }

        impl<'de> ::serde::de::Deserialize<'de> for $e {
            fn deserialize<D>(d: D) -> Result<Self, D::Error>
                    where D: ::serde::Deserializer<'de> {
                use serde::de::{Error, Visitor};
                use std::marker::PhantomData;
                use std::fmt;
                struct StringOnly<T>(PhantomData<T>);
                impl<'de, T> Visitor<'de> for StringOnly<T>
                        where T: ::serde::Deserializer<'de> {
                    type Value = String;
                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("string")
                    }
                    fn visit_str<E>(self, value: &str) -> Result<String, E> {
                        Ok(String::from(value))
                    }
                }
                let s = d.deserialize_string(StringOnly::<D>(PhantomData))?;
                $(
                    if stringify!($x).eq_ignore_ascii_case(&s) {
                      return Ok($e::$x);
                    }
                )*
                static ALLOWED: &'static[&str] = &[$(stringify!($x),)*];
                Err(D::Error::unknown_variant(&s, ALLOWED))
            }
        }

        impl ::std::str::FromStr for $e {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                $(
                    if stringify!($x).eq_ignore_ascii_case(s) {
                        return Ok($e::$x);
                    }
                )*
                Err("Bad variant")
            }
        }

        impl ConfigType for $e {
            fn doc_hint() -> String {
                let mut variants = Vec::new();
                $(
                    variants.push(stringify!($x));
                )*
                format!("[{}]", variants.join("|"))
            }
        }
    };
}

macro_rules! configuration_option_enum{
    ($e:ident: $( $x:ident ),+ $(,)*) => {
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        pub enum $e {
            $( $x ),+
        }

        impl_enum_serialize_and_deserialize!($e, $( $x ),+);
    }
}

configuration_option_enum! { NewlineStyle:
    Windows, // \r\n
    Unix, // \n
    Native, // \r\n in Windows, \n on other platforms
}

configuration_option_enum! { BraceStyle:
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where clause, in which case force
    // the brace to the next line.
    SameLineWhere,
}

configuration_option_enum! { ControlBraceStyle:
    // K&R style, Rust community default
    AlwaysSameLine,
    // Stroustrup style
    ClosingNextLine,
    // Allman style
    AlwaysNextLine,
}

configuration_option_enum! { IndentStyle:
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
}

configuration_option_enum! { Density:
    // Fit as much on one line as possible.
    Compressed,
    // Use more lines.
    Tall,
    // Place every item on a separate line.
    Vertical,
}

configuration_option_enum! { TypeDensity:
    // No spaces around "=" and "+"
    Compressed,
    // Spaces around " = " and " + "
    Wide,
}

impl Density {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            Density::Compressed => ListTactic::Mixed,
            Density::Tall => ListTactic::HorizontalVertical,
            Density::Vertical => ListTactic::Vertical,
        }
    }
}

configuration_option_enum! { ReportTactic:
    Always,
    Unnumbered,
    Never,
}

configuration_option_enum! { WriteMode:
    // Backs the original file up and overwrites the original.
    Replace,
    // Overwrites original file without backup.
    Overwrite,
    // Writes the output to stdout.
    Display,
    // Writes the diff to stdout.
    Diff,
    // Displays how much of the input file was processed
    Coverage,
    // Unfancy stdout
    Checkstyle,
    // Output the changed lines (for internal value only)
    Modified,
    // Checks if a diff can be generated. If so, rustfmt outputs a diff and quits with exit code 1.
    // This option is designed to be run in CI where a non-zero exit signifies non-standard code
    // formatting.
    Check,
    // Rustfmt shouldn't output anything formatting-like (e.g., emit a help message).
    None,
}

configuration_option_enum! { Color:
    // Always use color, whether it is a piped or terminal output
    Always,
    // Never use color
    Never,
    // Automatically use color, if supported by terminal
    Auto,
}

configuration_option_enum! { Verbosity:
    // Emit more.
    Verbose,
    Normal,
    // Emit as little as possible.
    Quiet,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct WidthHeuristics {
    // Maximum width of the args of a function call before falling back
    // to vertical formatting.
    pub fn_call_width: usize,
    // Maximum width in the body of a struct lit before falling back to
    // vertical formatting.
    pub struct_lit_width: usize,
    // Maximum width in the body of a struct variant before falling back
    // to vertical formatting.
    pub struct_variant_width: usize,
    // Maximum width of an array literal before falling back to vertical
    // formatting.
    pub array_width: usize,
    // Maximum length of a chain to fit on a single line.
    pub chain_width: usize,
    // Maximum line length for single line if-else expressions. A value
    // of zero means always break if-else expressions.
    pub single_line_if_else_max_width: usize,
}

impl WidthHeuristics {
    // Using this WidthHeuristics means we ignore heuristics.
    pub fn null() -> WidthHeuristics {
        WidthHeuristics {
            fn_call_width: usize::max_value(),
            struct_lit_width: 0,
            struct_variant_width: 0,
            array_width: usize::max_value(),
            chain_width: usize::max_value(),
            single_line_if_else_max_width: 0,
        }
    }

    // scale the default WidthHeuristics according to max_width
    pub fn scaled(max_width: usize) -> WidthHeuristics {
        const DEFAULT_MAX_WIDTH: usize = 100;
        let max_width_ratio = if max_width > DEFAULT_MAX_WIDTH {
            let ratio = max_width as f32 / DEFAULT_MAX_WIDTH as f32;
            // round to the closest 0.1
            (ratio * 10.0).round() / 10.0
        } else {
            1.0
        };
        WidthHeuristics {
            fn_call_width: (60.0 * max_width_ratio).round() as usize,
            struct_lit_width: (18.0 * max_width_ratio).round() as usize,
            struct_variant_width: (35.0 * max_width_ratio).round() as usize,
            array_width: (60.0 * max_width_ratio).round() as usize,
            chain_width: (60.0 * max_width_ratio).round() as usize,
            single_line_if_else_max_width: (50.0 * max_width_ratio).round() as usize,
        }
    }
}

impl ::std::str::FromStr for WidthHeuristics {
    type Err = &'static str;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err("WidthHeuristics is not parsable")
    }
}

impl Default for WriteMode {
    fn default() -> WriteMode {
        WriteMode::Overwrite
    }
}

/// A set of directories, files and modules that rustfmt should ignore.
#[derive(Default, Deserialize, Serialize, Clone, Debug)]
pub struct IgnoreList(HashSet<PathBuf>);

impl IgnoreList {
    pub fn add_prefix(&mut self, dir: &Path) {
        self.0 = self
            .0
            .iter()
            .map(|s| {
                if s.has_root() {
                    s.clone()
                } else {
                    let mut path = PathBuf::from(dir);
                    path.push(s);
                    path
                }
            })
            .collect();
    }

    fn skip_file_inner(&self, file: &Path) -> bool {
        self.0.iter().any(|path| file.starts_with(path))
    }

    pub fn skip_file(&self, file: &FileName) -> bool {
        if let FileName::Real(ref path) = file {
            self.skip_file_inner(path)
        } else {
            false
        }
    }
}

impl ::std::str::FromStr for IgnoreList {
    type Err = &'static str;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err("IgnoreList is not parsable")
    }
}

/// Parsed command line options.
#[derive(Clone, Debug, Default)]
pub struct CliOptions {
    skip_children: Option<bool>,
    quiet: bool,
    verbose: bool,
    verbose_diff: bool,
    pub(super) config_path: Option<PathBuf>,
    write_mode: Option<WriteMode>,
    color: Option<Color>,
    file_lines: FileLines, // Default is all lines in all files.
    unstable_features: bool,
    error_on_unformatted: Option<bool>,
}

impl CliOptions {
    pub fn from_matches(matches: &Matches) -> FmtResult<CliOptions> {
        let mut options = CliOptions::default();
        options.verbose = matches.opt_present("verbose");
        options.quiet = matches.opt_present("quiet");
        if options.verbose && options.quiet {
            return Err(format_err!("Can't use both `--verbose` and `--quiet`"));
        }
        options.verbose_diff = matches.opt_present("verbose-diff");

        let unstable_features = matches.opt_present("unstable-features");
        let rust_nightly = option_env!("CFG_RELEASE_CHANNEL")
            .map(|c| c == "nightly")
            .unwrap_or(false);
        if unstable_features && !rust_nightly {
            return Err(format_err!(
                "Unstable features are only available on Nightly channel"
            ));
        } else {
            options.unstable_features = unstable_features;
        }

        options.config_path = matches.opt_str("config-path").map(PathBuf::from);

        if let Some(ref write_mode) = matches.opt_str("write-mode") {
            if let Ok(write_mode) = WriteMode::from_str(write_mode) {
                options.write_mode = Some(write_mode);
            } else {
                return Err(format_err!(
                    "Invalid write-mode: {}, expected one of {}",
                    write_mode,
                    WRITE_MODE_LIST
                ));
            }
        }

        if let Some(ref color) = matches.opt_str("color") {
            match Color::from_str(color) {
                Ok(color) => options.color = Some(color),
                _ => return Err(format_err!("Invalid color: {}", color)),
            }
        }

        if let Some(ref file_lines) = matches.opt_str("file-lines") {
            options.file_lines = file_lines.parse().map_err(err_msg)?;
        }

        if matches.opt_present("skip-children") {
            options.skip_children = Some(true);
        }
        if matches.opt_present("error-on-unformatted") {
            options.error_on_unformatted = Some(true);
        }

        Ok(options)
    }

    pub fn apply_to(self, config: &mut Config) {
        if self.verbose {
            config.set().verbose(Verbosity::Verbose);
        } else if self.quiet {
            config.set().verbose(Verbosity::Quiet);
        } else {
            config.set().verbose(Verbosity::Normal);
        }
        config.set().verbose_diff(self.verbose_diff);
        config.set().file_lines(self.file_lines);
        config.set().unstable_features(self.unstable_features);
        if let Some(skip_children) = self.skip_children {
            config.set().skip_children(skip_children);
        }
        if let Some(error_on_unformatted) = self.error_on_unformatted {
            config.set().error_on_unformatted(error_on_unformatted);
        }
        if let Some(write_mode) = self.write_mode {
            config.set().write_mode(write_mode);
        }
        if let Some(color) = self.color {
            config.set().color(color);
        }
    }

    pub fn verify_file_lines(&self, files: &[PathBuf]) {
        for f in self.file_lines.files() {
            match *f {
                FileName::Real(ref f) if files.contains(f) => {}
                FileName::Real(_) => {
                    eprintln!("Warning: Extra file listed in file_lines option '{}'", f)
                }
                _ => eprintln!("Warning: Not a file '{}'", f),
            }
        }
    }
}
