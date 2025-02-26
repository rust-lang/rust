#![allow(unused_imports)]

use std::collections::{HashSet, hash_set};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use rustfmt_config_proc_macro::config_type;
use serde::de::{SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::config::Config;
use crate::config::file_lines::FileLines;
use crate::config::lists::*;
use crate::config::macro_names::MacroSelectors;

#[config_type]
pub enum NewlineStyle {
    /// Auto-detect based on the raw source input.
    Auto,
    /// Force CRLF (`\r\n`).
    Windows,
    /// Force CR (`\n`).
    Unix,
    /// `\r\n` in Windows, `\n` on other platforms.
    Native,
}

#[config_type]
/// Where to put the opening brace of items (`fn`, `impl`, etc.).
pub enum BraceStyle {
    /// Put the opening brace on the next line.
    AlwaysNextLine,
    /// Put the opening brace on the same line, if possible.
    PreferSameLine,
    /// Prefer the same line except where there is a where-clause, in which
    /// case force the brace to be put on the next line.
    SameLineWhere,
}

#[config_type]
/// Where to put the opening brace of conditional expressions (`if`, `match`, etc.).
pub enum ControlBraceStyle {
    /// K&R style, Rust community default
    AlwaysSameLine,
    /// Stroustrup style
    ClosingNextLine,
    /// Allman style
    AlwaysNextLine,
}

#[config_type]
/// How to indent.
pub enum IndentStyle {
    /// First line on the same line as the opening brace, all lines aligned with
    /// the first line.
    Visual,
    /// First line is on a new line and all lines align with **block** indent.
    Block,
}

#[config_type]
/// How to place a list-like items.
/// FIXME: Issue-3581: this should be renamed to ItemsLayout when publishing 2.0
pub enum Density {
    /// Fit as much on one line as possible.
    Compressed,
    /// Items are placed horizontally if sufficient space, vertically otherwise.
    Tall,
    /// Place every item on a separate line.
    Vertical,
}

#[config_type]
/// Spacing around type combinators.
pub enum TypeDensity {
    /// No spaces around "=" and "+"
    Compressed,
    /// Spaces around " = " and " + "
    Wide,
}

#[config_type]
/// Heuristic settings that can be used to simply
/// the configuration of the granular width configurations
/// like `struct_lit_width`, `array_width`, etc.
pub enum Heuristics {
    /// Turn off any heuristics
    Off,
    /// Turn on max heuristics
    Max,
    /// Use scaled values based on the value of `max_width`
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

#[config_type]
/// Configuration for import groups, i.e. sets of imports separated by newlines.
pub enum GroupImportsTactic {
    /// Keep groups as they are.
    Preserve,
    /// Discard existing groups, and create new groups for
    ///  1. `std` / `core` / `alloc` imports
    ///  2. other imports
    ///  3. `self` / `crate` / `super` imports
    StdExternalCrate,
    /// Discard existing groups, and create a single group for everything
    One,
}

#[config_type]
/// How to merge imports.
pub enum ImportGranularity {
    /// Do not merge imports.
    Preserve,
    /// Use one `use` statement per crate.
    Crate,
    /// Use one `use` statement per module.
    Module,
    /// Use one `use` statement per imported item.
    Item,
    /// Use one `use` statement including all items.
    One,
}

/// Controls how rustfmt should handle case in hexadecimal literals.
#[config_type]
pub enum HexLiteralCase {
    /// Leave the literal as-is
    Preserve,
    /// Ensure all literals use uppercase lettering
    Upper,
    /// Ensure all literals use lowercase lettering
    Lower,
}

#[config_type]
pub enum ReportTactic {
    Always,
    Unnumbered,
    Never,
}

/// What Rustfmt should emit. Mostly corresponds to the `--emit` command line
/// option.
#[config_type]
pub enum EmitMode {
    /// Emits to files.
    Files,
    /// Writes the output to stdout.
    Stdout,
    /// Displays how much of the input file was processed
    Coverage,
    /// Unfancy stdout
    Checkstyle,
    /// Writes the resulting diffs in a JSON format. Returns an empty array
    /// `[]` if there were no diffs.
    Json,
    /// Output the changed lines (for internal value only)
    ModifiedLines,
    /// Checks if a diff can be generated. If so, rustfmt outputs a diff and
    /// quits with exit code 1.
    /// This option is designed to be run in CI where a non-zero exit signifies
    /// non-standard code formatting. Used for `--check`.
    Diff,
}

/// Client-preference for coloured output.
#[config_type]
pub enum Color {
    /// Always use color, whether it is a piped or terminal output
    Always,
    /// Never use color
    Never,
    /// Automatically use color, if supported by terminal
    Auto,
}

#[config_type]
/// rustfmt format style version.
pub enum Version {
    /// 1.x.y. When specified, rustfmt will format in the same style as 1.0.0.
    One,
    /// 2.x.y. When specified, rustfmt will format in the latest style.
    Two,
}

impl Color {
    /// Whether we should use a coloured terminal.
    pub fn use_colored_tty(self) -> bool {
        match self {
            Color::Always | Color::Auto => true,
            Color::Never => false,
        }
    }
}

/// How chatty should Rustfmt be?
#[config_type]
pub enum Verbosity {
    /// Emit more.
    Verbose,
    /// Default.
    Normal,
    /// Emit as little as possible.
    Quiet,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct WidthHeuristics {
    // Maximum width of the args of a function call before falling back
    // to vertical formatting.
    pub(crate) fn_call_width: usize,
    // Maximum width of the args of a function-like attributes before falling
    // back to vertical formatting.
    pub(crate) attr_fn_like_width: usize,
    // Maximum width in the body of a struct lit before falling back to
    // vertical formatting.
    pub(crate) struct_lit_width: usize,
    // Maximum width in the body of a struct variant before falling back
    // to vertical formatting.
    pub(crate) struct_variant_width: usize,
    // Maximum width of an array literal before falling back to vertical
    // formatting.
    pub(crate) array_width: usize,
    // Maximum length of a chain to fit on a single line.
    pub(crate) chain_width: usize,
    // Maximum line length for single line if-else expressions. A value
    // of zero means always break if-else expressions.
    pub(crate) single_line_if_else_max_width: usize,
    // Maximum line length for single line let-else statements. A value of zero means
    // always format the divergent `else` block over multiple lines.
    pub(crate) single_line_let_else_max_width: usize,
}

impl fmt::Display for WidthHeuristics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl WidthHeuristics {
    // Using this WidthHeuristics means we ignore heuristics.
    pub fn null() -> WidthHeuristics {
        WidthHeuristics {
            fn_call_width: usize::MAX,
            attr_fn_like_width: usize::MAX,
            struct_lit_width: 0,
            struct_variant_width: 0,
            array_width: usize::MAX,
            chain_width: usize::MAX,
            single_line_if_else_max_width: 0,
            single_line_let_else_max_width: 0,
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
            single_line_let_else_max_width: max_width,
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
            single_line_let_else_max_width: (50.0 * max_width_ratio).round() as usize,
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
#[derive(Default, Clone, Debug, PartialEq)]
pub struct IgnoreList {
    /// A set of path specified in rustfmt.toml.
    path_set: HashSet<PathBuf>,
    /// A path to rustfmt.toml.
    rustfmt_toml_path: PathBuf,
}

impl fmt::Display for IgnoreList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.path_set
                .iter()
                .format_with(", ", |path, f| f(&format_args!(
                    "{}",
                    path.to_string_lossy()
                )))
        )
    }
}

impl Serialize for IgnoreList {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.path_set.len()))?;
        for e in &self.path_set {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
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

impl FromStr for IgnoreList {
    type Err = &'static str;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err("IgnoreList is not parsable")
    }
}

/// Maps client-supplied options to Rustfmt's internals, mostly overriding
/// values in a config with values from the command line.
pub trait CliOptions {
    fn apply_to(self, config: &mut Config);

    /// It is ok if the returned path doesn't exist or is not canonicalized
    /// (i.e. the callers are expected to handle such cases).
    fn config_path(&self) -> Option<&Path>;
    fn edition(&self) -> Option<Edition>;
    fn style_edition(&self) -> Option<StyleEdition>;
    fn version(&self) -> Option<Version>;
}

/// The edition of the syntax and semantics of code (RFC 2052).
#[config_type]
pub enum Edition {
    #[value = "2015"]
    #[doc_hint = "2015"]
    /// Edition 2015.
    Edition2015,
    #[value = "2018"]
    #[doc_hint = "2018"]
    /// Edition 2018.
    Edition2018,
    #[value = "2021"]
    #[doc_hint = "2021"]
    /// Edition 2021.
    Edition2021,
    #[value = "2024"]
    #[doc_hint = "2024"]
    /// Edition 2024.
    Edition2024,
}

impl Default for Edition {
    fn default() -> Edition {
        Edition::Edition2015
    }
}

impl From<Edition> for rustc_span::edition::Edition {
    fn from(edition: Edition) -> Self {
        match edition {
            Edition::Edition2015 => Self::Edition2015,
            Edition::Edition2018 => Self::Edition2018,
            Edition::Edition2021 => Self::Edition2021,
            Edition::Edition2024 => Self::Edition2024,
        }
    }
}

impl From<Edition> for StyleEdition {
    fn from(edition: Edition) -> Self {
        match edition {
            Edition::Edition2015 => StyleEdition::Edition2015,
            Edition::Edition2018 => StyleEdition::Edition2018,
            Edition::Edition2021 => StyleEdition::Edition2021,
            Edition::Edition2024 => StyleEdition::Edition2024,
        }
    }
}

impl PartialOrd for Edition {
    fn partial_cmp(&self, other: &Edition) -> Option<std::cmp::Ordering> {
        rustc_span::edition::Edition::partial_cmp(&(*self).into(), &(*other).into())
    }
}

/// Controls how rustfmt should handle leading pipes on match arms.
#[config_type]
pub enum MatchArmLeadingPipe {
    /// Place leading pipes on all match arms
    Always,
    /// Never emit leading pipes on match arms
    Never,
    /// Preserve any existing leading pipes
    Preserve,
}

/// Defines the default values for each config according to the edition of the
/// [Style Guide] as per [RFC 3338]. Rustfmt output may differ between Style editions.
///
/// [Style Guide]: https://doc.rust-lang.org/nightly/style-guide/
/// [RFC 3338]: https://rust-lang.github.io/rfcs/3338-style-evolution.html
#[config_type]
pub enum StyleEdition {
    #[value = "2015"]
    #[doc_hint = "2015"]
    /// [Edition 2015]()
    Edition2015,
    #[value = "2018"]
    #[doc_hint = "2018"]
    /// [Edition 2018]()
    Edition2018,
    #[value = "2021"]
    #[doc_hint = "2021"]
    /// [Edition 2021]()
    Edition2021,
    #[value = "2024"]
    #[doc_hint = "2024"]
    /// [Edition 2024]().
    Edition2024,
}

impl From<StyleEdition> for rustc_span::edition::Edition {
    fn from(edition: StyleEdition) -> Self {
        match edition {
            StyleEdition::Edition2015 => Self::Edition2015,
            StyleEdition::Edition2018 => Self::Edition2018,
            StyleEdition::Edition2021 => Self::Edition2021,
            StyleEdition::Edition2024 => Self::Edition2024,
        }
    }
}

impl PartialOrd for StyleEdition {
    fn partial_cmp(&self, other: &StyleEdition) -> Option<std::cmp::Ordering> {
        rustc_span::edition::Edition::partial_cmp(&(*self).into(), &(*other).into())
    }
}

/// Defines unit structs to implement `StyleEditionDefault` for.
#[macro_export]
macro_rules! config_option_with_style_edition_default {
    ($name:ident, $config_ty:ty, _ => $default:expr) => {
        #[allow(unreachable_pub)]
        pub struct $name;
        $crate::style_edition_default!($name, $config_ty, _ => $default);
    };
    ($name:ident, $config_ty:ty, Edition2024 => $default_2024:expr, _ => $default_2015:expr) => {
        pub struct $name;
        $crate::style_edition_default!(
            $name,
            $config_ty,
            Edition2024 => $default_2024,
            _ => $default_2015
        );
    };
    (
        $($name:ident, $config_ty:ty, $(Edition2024 => $default_2024:expr,)? _ => $default:expr);*
        $(;)*
    ) => {
        $(
            config_option_with_style_edition_default!(
                $name, $config_ty, $(Edition2024 => $default_2024,)? _ => $default
            );
        )*
    };
}

// TODO(ytmimi) Some of the configuration values have a `Config` suffix, while others don't.
// I chose to add a `Config` suffix in cases where a type for the config option was already
// defined. For example, `NewlineStyle` and `NewlineStyleConfig`. There was some discussion
// about using the `Config` suffix more consistently.
config_option_with_style_edition_default!(
    // Fundamental stuff
    MaxWidth, usize, _ => 100;
    HardTabs, bool, _ => false;
    TabSpaces, usize, _ => 4;
    NewlineStyleConfig, NewlineStyle, _ => NewlineStyle::Auto;
    IndentStyleConfig, IndentStyle, _ => IndentStyle::Block;

    // Width Heuristics
    UseSmallHeuristics, Heuristics, _ => Heuristics::Default;
    WidthHeuristicsConfig, WidthHeuristics, _ => WidthHeuristics::scaled(100);
    FnCallWidth, usize, _ => 60;
    AttrFnLikeWidth, usize, _ => 70;
    StructLitWidth, usize, _ => 18;
    StructVariantWidth, usize, _ => 35;
    ArrayWidth, usize, _ => 60;
    ChainWidth, usize, _ => 60;
    SingleLineIfElseMaxWidth, usize, _ => 50;
    SingleLineLetElseMaxWidth, usize, _ => 50;

    // Comments. macros, and strings
    WrapComments, bool, _ => false;
    FormatCodeInDocComments, bool, _ => false;
    DocCommentCodeBlockWidth, usize, _ => 100;
    CommentWidth, usize, _ => 80;
    NormalizeComments, bool, _ => false;
    NormalizeDocAttributes, bool, _ => false;
    FormatStrings, bool, _ => false;
    FormatMacroMatchers, bool, _ => false;
    FormatMacroBodies, bool, _ => true;
    SkipMacroInvocations, MacroSelectors, _ => MacroSelectors::default();
    HexLiteralCaseConfig, HexLiteralCase, _ => HexLiteralCase::Preserve;

    // Single line expressions and items
    EmptyItemSingleLine, bool, _ => true;
    StructLitSingleLine, bool, _ => true;
    FnSingleLine, bool, _ => false;
    WhereSingleLine, bool, _ => false;

    // Imports
    ImportsIndent, IndentStyle, _ => IndentStyle::Block;
    ImportsLayout, ListTactic, _ => ListTactic::Mixed;
    ImportsGranularityConfig, ImportGranularity, _ => ImportGranularity::Preserve;
    GroupImportsTacticConfig, GroupImportsTactic, _ => GroupImportsTactic::Preserve;
    MergeImports, bool, _ => false;

    // Ordering
    ReorderImports, bool, _ => true;
    ReorderModules, bool, _ => true;
    ReorderImplItems, bool, _ => false;

    // Spaces around punctuation
    TypePunctuationDensity, TypeDensity, _ => TypeDensity::Wide;
    SpaceBeforeColon, bool, _ => false;
    SpaceAfterColon, bool, _ => true;
    SpacesAroundRanges, bool, _ => false;
    BinopSeparator, SeparatorPlace, _ => SeparatorPlace::Front;

    // Misc.
    RemoveNestedParens, bool, _ => true;
    CombineControlExpr, bool, _ => true;
    ShortArrayElementWidthThreshold, usize, _ => 10;
    OverflowDelimitedExpr, bool, _ => false;
    StructFieldAlignThreshold, usize, _ => 0;
    EnumDiscrimAlignThreshold, usize, _ => 0;
    MatchArmBlocks, bool, _ => true;
    MatchArmLeadingPipeConfig, MatchArmLeadingPipe, _ => MatchArmLeadingPipe::Never;
    ForceMultilineBlocks, bool, _ => false;
    FnArgsLayout, Density, _ => Density::Tall;
    FnParamsLayout, Density, _ => Density::Tall;
    BraceStyleConfig, BraceStyle, _ => BraceStyle::SameLineWhere;
    ControlBraceStyleConfig, ControlBraceStyle, _ => ControlBraceStyle::AlwaysSameLine;
    TrailingSemicolon, bool, _ => true;
    TrailingComma, SeparatorTactic, _ => SeparatorTactic::Vertical;
    MatchBlockTrailingComma, bool, _ => false;
    BlankLinesUpperBound, usize, _ => 1;
    BlankLinesLowerBound, usize, _ => 0;
    EditionConfig, Edition, _ => Edition::Edition2015;
    StyleEditionConfig, StyleEdition,
        Edition2024 => StyleEdition::Edition2024, _ => StyleEdition::Edition2015;
    VersionConfig, Version, Edition2024 => Version::Two, _ => Version::One;
    InlineAttributeWidth, usize, _ => 0;
    FormatGeneratedFiles, bool, _ => true;
    GeneratedMarkerLineSearchLimit, usize, _ => 5;

    // Options that can change the source code beyond whitespace/blocks (somewhat linty things)
    MergeDerives, bool, _ => true;
    UseTryShorthand, bool, _ => false;
    UseFieldInitShorthand, bool, _ => false;
    ForceExplicitAbi, bool, _ => true;
    CondenseWildcardSuffixes, bool, _ => false;

    // Control options (changes the operation of rustfmt, rather than the formatting)
    ColorConfig, Color, _ => Color::Auto;
    RequiredVersion, String, _ => env!("CARGO_PKG_VERSION").to_owned();
    UnstableFeatures, bool, _ => false;
    DisableAllFormatting, bool, _ => false;
    SkipChildren, bool, _ => false;
    HideParseErrors, bool, _ => false;
    ShowParseErrors, bool, _ => true;
    ErrorOnLineOverflow, bool, _ => false;
    ErrorOnUnformatted, bool, _ => false;
    Ignore, IgnoreList, _ => IgnoreList::default();

    // Not user-facing
    Verbose, Verbosity, _ => Verbosity::Normal;
    FileLinesConfig, FileLines, _ => FileLines::all();
    EmitModeConfig, EmitMode, _ => EmitMode::Files;
    MakeBackup, bool, _ => false;
    PrintMisformattedFileNames, bool, _ => false;
);
