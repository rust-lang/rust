use std::collections::{hash_set, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use atty;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};

use crate::config::config_type::ConfigType;
use crate::config::lists::*;
use crate::config::Config;

/// Macro that will stringify the enum variants or a provided textual repr
#[macro_export]
macro_rules! configuration_option_enum_stringify {
    ($variant:ident) => {
        stringify!($variant)
    };

    ($_variant:ident: $value:expr) => {
        stringify!($value)
    };
}

/// Macro for deriving implementations of Serialize/Deserialize for enums
#[macro_export]
macro_rules! impl_enum_serialize_and_deserialize {
    ( $e:ident, $( $variant:ident $(: $value:expr)* ),* ) => {
        impl ::serde::ser::Serialize for $e {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: ::serde::ser::Serializer
            {
                use serde::ser::Error;

                // We don't know whether the user of the macro has given us all options.
                #[allow(unreachable_patterns)]
                match *self {
                    $(
                        $e::$variant => serializer.serialize_str(
                            configuration_option_enum_stringify!($variant $(: $value)*)
                        ),
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
                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str("string")
                    }
                    fn visit_str<E>(self, value: &str) -> Result<String, E> {
                        Ok(String::from(value))
                    }
                }
                let s = d.deserialize_string(StringOnly::<D>(PhantomData))?;
                $(
                    if configuration_option_enum_stringify!($variant $(: $value)*)
                        .eq_ignore_ascii_case(&s) {
                      return Ok($e::$variant);
                    }
                )*
                static ALLOWED: &'static[&str] = &[
                    $(configuration_option_enum_stringify!($variant $(: $value)*),)*];
                Err(D::Error::unknown_variant(&s, ALLOWED))
            }
        }

        impl ::std::str::FromStr for $e {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                $(
                    if configuration_option_enum_stringify!($variant $(: $value)*)
                        .eq_ignore_ascii_case(s) {
                        return Ok($e::$variant);
                    }
                )*
                Err("Bad variant")
            }
        }

        impl ConfigType for $e {
            fn doc_hint() -> String {
                let mut variants = Vec::new();
                $(
                    variants.push(
                        configuration_option_enum_stringify!($variant $(: $value)*)
                    );
                )*
                format!("[{}]", variants.join("|"))
            }
        }
    };
}

macro_rules! configuration_option_enum {
    ($e:ident: $( $name:ident $(: $value:expr)* ),+ $(,)*) => (
        #[derive(Copy, Clone, Eq, PartialEq)]
        pub enum $e {
            $( $name ),+
        }

        impl ::std::fmt::Debug for $e {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(match self {
                    $(
                        $e::$name => configuration_option_enum_stringify!($name $(: $value)*),
                    )+
                })
            }
        }

        impl_enum_serialize_and_deserialize!($e, $( $name $(: $value)* ),+);
    );
}

configuration_option_enum! { NewlineStyle:
    Auto, // Auto-detect based on the raw source input
    Windows, // \r\n
    Unix, // \n
    Native, // \r\n in Windows, \n on other platforms
}

impl NewlineStyle {
    fn auto_detect(raw_input_text: &str) -> NewlineStyle {
        if let Some(pos) = raw_input_text.find('\n') {
            let pos = pos.saturating_sub(1);
            if let Some('\r') = raw_input_text.chars().nth(pos) {
                NewlineStyle::Windows
            } else {
                NewlineStyle::Unix
            }
        } else {
            NewlineStyle::Native
        }
    }

    fn native() -> NewlineStyle {
        if cfg!(windows) {
            NewlineStyle::Windows
        } else {
            NewlineStyle::Unix
        }
    }

    /// Apply this newline style to the formatted text. When the style is set
    /// to `Auto`, the `raw_input_text` is used to detect the existing line
    /// endings.
    ///
    /// If the style is set to `Auto` and `raw_input_text` contains no
    /// newlines, the `Native` style will be used.
    pub(crate) fn apply(self, formatted_text: &mut String, raw_input_text: &str) {
        use crate::NewlineStyle::*;
        let mut style = self;
        if style == Auto {
            style = Self::auto_detect(raw_input_text);
        }
        if style == Native {
            style = Self::native();
        }
        match style {
            Windows => {
                let mut transformed = String::with_capacity(2 * formatted_text.capacity());
                for c in formatted_text.chars() {
                    match c {
                        '\n' => transformed.push_str("\r\n"),
                        '\r' => continue,
                        c => transformed.push(c),
                    }
                }
                *formatted_text = transformed;
            }
            Unix => return,
            Native => unreachable!("NewlineStyle::Native"),
            Auto => unreachable!("NewlineStyle::Auto"),
        }
    }
}

configuration_option_enum! { BraceStyle:
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where-clause, in which case force
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

configuration_option_enum! { Heuristics:
    // Turn off any heuristics
    Off,
    // Turn on max heuristics
    Max,
    // Use Rustfmt's defaults
    Default,
}

impl Density {
    pub fn to_list_tactic(self, len: usize) -> ListTactic {
        match self {
            Density::Compressed => ListTactic::Mixed,
            Density::Tall => ListTactic::HorizontalVertical,
            Density::Vertical if len == 1 => ListTactic::Horizontal,
            Density::Vertical => ListTactic::Vertical,
        }
    }
}

configuration_option_enum! { ReportTactic:
    Always,
    Unnumbered,
    Never,
}

// What Rustfmt should emit. Mostly corresponds to the `--emit` command line
// option.
configuration_option_enum! { EmitMode:
    // Emits to files.
    Files,
    // Writes the output to stdout.
    Stdout,
    // Displays how much of the input file was processed
    Coverage,
    // Unfancy stdout
    Checkstyle,
    // Output the changed lines (for internal value only)
    ModifiedLines,
    // Checks if a diff can be generated. If so, rustfmt outputs a diff and quits with exit code 1.
    // This option is designed to be run in CI where a non-zero exit signifies non-standard code
    // formatting. Used for `--check`.
    Diff,
}

// Client-preference for coloured output.
configuration_option_enum! { Color:
    // Always use color, whether it is a piped or terminal output
    Always,
    // Never use color
    Never,
    // Automatically use color, if supported by terminal
    Auto,
}

configuration_option_enum! { Version:
    // 1.x.y
    One,
    // 2.x.y
    Two,
}

impl Color {
    /// Whether we should use a coloured terminal.
    pub fn use_colored_tty(self) -> bool {
        match self {
            Color::Always => true,
            Color::Never => false,
            Color::Auto => atty::is(atty::Stream::Stdout),
        }
    }
}

// How chatty should Rustfmt be?
configuration_option_enum! { Verbosity:
    // Emit more.
    Verbose,
    Normal,
    // Emit as little as possible.
    Quiet,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct WidthHeuristics {
    // Maximum width of the args of a function call before falling back
    // to vertical formatting.
    pub fn_call_width: usize,
    // Maximum width of the args of a function-like attributes before falling
    // back to vertical formatting.
    pub attr_fn_like_width: usize,
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
            attr_fn_like_width: usize::max_value(),
            struct_lit_width: 0,
            struct_variant_width: 0,
            array_width: usize::max_value(),
            chain_width: usize::max_value(),
            single_line_if_else_max_width: 0,
        }
    }

    pub fn set(max_width: usize) -> WidthHeuristics {
        WidthHeuristics {
            fn_call_width: max_width,
            attr_fn_like_width: max_width,
            struct_lit_width: max_width,
            struct_variant_width: max_width,
            array_width: max_width,
            chain_width: max_width,
            single_line_if_else_max_width: max_width,
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
            attr_fn_like_width: (70.0 * max_width_ratio).round() as usize,
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

impl Default for EmitMode {
    fn default() -> EmitMode {
        EmitMode::Files
    }
}

/// A set of directories, files and modules that rustfmt should ignore.
#[derive(Default, Serialize, Clone, Debug, PartialEq)]
pub struct IgnoreList {
    /// A set of path specified in rustfmt.toml.
    #[serde(flatten)]
    path_set: HashSet<PathBuf>,
    /// A path to rustfmt.toml.
    #[serde(skip_serializing)]
    rustfmt_toml_path: PathBuf,
}

impl<'de> Deserialize<'de> for IgnoreList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HashSetVisitor;
        impl<'v> Visitor<'v> for HashSetVisitor {
            type Value = HashSet<PathBuf>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of path")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'v>,
            {
                let mut path_set = HashSet::new();
                while let Some(elem) = seq.next_element()? {
                    path_set.insert(elem);
                }
                Ok(path_set)
            }
        }
        Ok(IgnoreList {
            path_set: deserializer.deserialize_seq(HashSetVisitor)?,
            rustfmt_toml_path: PathBuf::new(),
        })
    }
}

impl<'a> IntoIterator for &'a IgnoreList {
    type Item = &'a PathBuf;
    type IntoIter = hash_set::Iter<'a, PathBuf>;

    fn into_iter(self) -> Self::IntoIter {
        self.path_set.iter()
    }
}

impl IgnoreList {
    pub fn add_prefix(&mut self, dir: &Path) {
        self.rustfmt_toml_path = dir.to_path_buf();
    }

    pub fn rustfmt_toml_path(&self) -> &Path {
        &self.rustfmt_toml_path
    }
}

impl ::std::str::FromStr for IgnoreList {
    type Err = &'static str;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err("IgnoreList is not parsable")
    }
}

/// Maps client-supplied options to Rustfmt's internals, mostly overriding
/// values in a config with values from the command line.
pub trait CliOptions {
    fn apply_to(self, config: &mut Config);
    fn config_path(&self) -> Option<&Path>;
}

// The edition of the compiler (RFC 2052)
configuration_option_enum! { Edition:
    Edition2015: 2015,
    Edition2018: 2018,
}

impl Default for Edition {
    fn default() -> Edition {
        Edition::Edition2015
    }
}

impl Edition {
    pub(crate) fn to_libsyntax_pos_edition(self) -> syntax_pos::edition::Edition {
        match self {
            Edition::Edition2015 => syntax_pos::edition::Edition::Edition2015,
            Edition::Edition2018 => syntax_pos::edition::Edition::Edition2018,
        }
    }
}

#[test]
fn test_newline_style_auto_detect() {
    let lf = "One\nTwo\nThree";
    let crlf = "One\r\nTwo\r\nThree";
    let none = "One Two Three";

    assert_eq!(NewlineStyle::Unix, NewlineStyle::auto_detect(lf));
    assert_eq!(NewlineStyle::Windows, NewlineStyle::auto_detect(crlf));
    assert_eq!(NewlineStyle::Native, NewlineStyle::auto_detect(none));
}

#[test]
fn test_newline_style_auto_apply() {
    let auto = NewlineStyle::Auto;

    let formatted_text = "One\nTwo\nThree";
    let raw_input_text = "One\nTwo\nThree";

    let mut out = String::from(formatted_text);
    auto.apply(&mut out, raw_input_text);
    assert_eq!("One\nTwo\nThree", &out, "auto should detect 'lf'");

    let formatted_text = "One\nTwo\nThree";
    let raw_input_text = "One\r\nTwo\r\nThree";

    let mut out = String::from(formatted_text);
    auto.apply(&mut out, raw_input_text);
    assert_eq!("One\r\nTwo\r\nThree", &out, "auto should detect 'crlf'");

    #[cfg(not(windows))]
    {
        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One Two Three";

        let mut out = String::from(formatted_text);
        auto.apply(&mut out, raw_input_text);
        assert_eq!(
            "One\nTwo\nThree", &out,
            "auto-native-unix should detect 'lf'"
        );
    }

    #[cfg(windows)]
    {
        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One Two Three";

        let mut out = String::from(formatted_text);
        auto.apply(&mut out, raw_input_text);
        assert_eq!(
            "One\r\nTwo\r\nThree", &out,
            "auto-native-windows should detect 'crlf'"
        );
    }
}
