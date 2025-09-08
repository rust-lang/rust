use crate::ClippyConfiguration;
use crate::types::{
    DisallowedPath, DisallowedPathWithoutReplacement, MacroMatcher, MatchLintBehaviour, PubUnderscoreFieldsBehaviour,
    Rename, SourceItemOrdering, SourceItemOrderingCategory, SourceItemOrderingModuleItemGroupings,
    SourceItemOrderingModuleItemKind, SourceItemOrderingTraitAssocItemKind, SourceItemOrderingTraitAssocItemKinds,
    SourceItemOrderingWithinModuleItemGroupings,
};
use clippy_utils::msrvs::Msrv;
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_session::Session;
use rustc_span::edit_distance::edit_distance;
use rustc_span::{BytePos, Pos, SourceFile, Span, SyntaxContext};
use serde::de::{IgnoredAny, IntoDeserializer, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::OnceLock;
use std::{cmp, env, fmt, fs, io};

#[rustfmt::skip]
const DEFAULT_DOC_VALID_IDENTS: &[&str] = &[
    "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
    "MHz", "GHz", "THz",
    "AccessKit",
    "CoAP", "CoreFoundation", "CoreGraphics", "CoreText",
    "DevOps",
    "Direct2D", "Direct3D", "DirectWrite", "DirectX",
    "ECMAScript",
    "GPLv2", "GPLv3",
    "GitHub", "GitLab",
    "IPv4", "IPv6",
    "InfiniBand", "RoCE",
    "ClojureScript", "CoffeeScript", "JavaScript", "PostScript", "PureScript", "TypeScript",
    "PowerPC", "WebAssembly",
    "NaN", "NaNs",
    "OAuth", "GraphQL",
    "OCaml",
    "OpenAL", "OpenDNS", "OpenGL", "OpenMP", "OpenSSH", "OpenSSL", "OpenStreetMap", "OpenTelemetry",
    "OpenType",
    "WebGL", "WebGL2", "WebGPU", "WebRTC", "WebSocket", "WebTransport",
    "WebP", "OpenExr", "YCbCr", "sRGB",
    "TensorFlow",
    "TrueType",
    "iOS", "macOS", "FreeBSD", "NetBSD", "OpenBSD", "NixOS",
    "TeX", "LaTeX", "BibTeX", "BibLaTeX",
    "MinGW",
    "CamelCase",
];
const DEFAULT_DISALLOWED_NAMES: &[&str] = &["foo", "baz", "quux"];
const DEFAULT_ALLOWED_IDENTS_BELOW_MIN_CHARS: &[&str] = &["i", "j", "x", "y", "z", "w", "n"];
const DEFAULT_ALLOWED_PREFIXES: &[&str] = &["to", "as", "into", "from", "try_into", "try_from"];
const DEFAULT_ALLOWED_TRAITS_WITH_RENAMED_PARAMS: &[&str] =
    &["core::convert::From", "core::convert::TryFrom", "core::str::FromStr"];
const DEFAULT_MODULE_ITEM_ORDERING_GROUPS: &[(&str, &[SourceItemOrderingModuleItemKind])] = {
    #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
    use SourceItemOrderingModuleItemKind::*;
    &[
        ("modules", &[ExternCrate, Mod, ForeignMod]),
        ("use", &[Use]),
        ("macros", &[Macro]),
        ("global_asm", &[GlobalAsm]),
        ("UPPER_SNAKE_CASE", &[Static, Const]),
        ("PascalCase", &[TyAlias, Enum, Struct, Union, Trait, TraitAlias, Impl]),
        ("lower_snake_case", &[Fn]),
    ]
};
const DEFAULT_TRAIT_ASSOC_ITEM_KINDS_ORDER: &[SourceItemOrderingTraitAssocItemKind] = {
    #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
    use SourceItemOrderingTraitAssocItemKind::*;
    &[Const, Type, Fn]
};
const DEFAULT_SOURCE_ITEM_ORDERING: &[SourceItemOrderingCategory] = {
    #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
    use SourceItemOrderingCategory::*;
    &[Enum, Impl, Module, Struct, Trait]
};

/// Conf with parse errors
#[derive(Default)]
struct TryConf {
    conf: Conf,
    value_spans: HashMap<String, Range<usize>>,
    errors: Vec<ConfError>,
    warnings: Vec<ConfError>,
}

impl TryConf {
    fn from_toml_error(file: &SourceFile, error: &toml::de::Error) -> Self {
        Self {
            conf: Conf::default(),
            value_spans: HashMap::default(),
            errors: vec![ConfError::from_toml(file, error)],
            warnings: vec![],
        }
    }
}

#[derive(Debug)]
struct ConfError {
    message: String,
    suggestion: Option<Suggestion>,
    span: Span,
}

impl ConfError {
    fn from_toml(file: &SourceFile, error: &toml::de::Error) -> Self {
        let span = error.span().unwrap_or(0..file.source_len.0 as usize);
        Self::spanned(file, error.message(), None, span)
    }

    fn spanned(
        file: &SourceFile,
        message: impl Into<String>,
        suggestion: Option<Suggestion>,
        span: Range<usize>,
    ) -> Self {
        Self {
            message: message.into(),
            suggestion,
            span: span_from_toml_range(file, span),
        }
    }
}

// Remove code tags and code behind '# 's, as they are not needed for the lint docs and --explain
pub fn sanitize_explanation(raw_docs: &str) -> String {
    // Remove tags and hidden code:
    let mut explanation = String::with_capacity(128);
    let mut in_code = false;
    for line in raw_docs.lines() {
        let line = line.strip_prefix(' ').unwrap_or(line);

        if let Some(lang) = line.strip_prefix("```") {
            let tag = lang.split_once(',').map_or(lang, |(left, _)| left);
            if !in_code && matches!(tag, "" | "rust" | "ignore" | "should_panic" | "no_run" | "compile_fail") {
                explanation += "```rust\n";
            } else {
                explanation += line;
                explanation.push('\n');
            }
            in_code = !in_code;
        } else if !(in_code && line.starts_with("# ")) {
            explanation += line;
            explanation.push('\n');
        }
    }

    explanation
}

macro_rules! wrap_option {
    () => {
        None
    };
    ($x:literal) => {
        Some($x)
    };
}

macro_rules! default_text {
    ($value:expr) => {{
        let mut text = String::new();
        $value.serialize(toml::ser::ValueSerializer::new(&mut text)).unwrap();
        text
    }};
    ($value:expr, $override:expr) => {
        $override.to_string()
    };
}

macro_rules! deserialize {
    ($map:expr, $ty:ty, $errors:expr, $file:expr) => {{
        let raw_value = $map.next_value::<toml::Spanned<toml::Value>>()?;
        let value_span = raw_value.span();
        let value = match <$ty>::deserialize(raw_value.into_inner()) {
            Err(e) => {
                $errors.push(ConfError::spanned(
                    $file,
                    e.to_string().replace('\n', " ").trim(),
                    None,
                    value_span,
                ));
                continue;
            },
            Ok(value) => value,
        };
        (value, value_span)
    }};

    ($map:expr, $ty:ty, $errors:expr, $file:expr, $replacements_allowed:expr) => {{
        let array = $map.next_value::<Vec<toml::Spanned<toml::Value>>>()?;
        let mut disallowed_paths_span = Range {
            start: usize::MAX,
            end: usize::MIN,
        };
        let mut disallowed_paths = Vec::new();
        for raw_value in array {
            let value_span = raw_value.span();
            let mut disallowed_path = match DisallowedPath::<$replacements_allowed>::deserialize(raw_value.into_inner())
            {
                Err(e) => {
                    $errors.push(ConfError::spanned(
                        $file,
                        e.to_string().replace('\n', " ").trim(),
                        None,
                        value_span,
                    ));
                    continue;
                },
                Ok(disallowed_path) => disallowed_path,
            };
            disallowed_paths_span = union(&disallowed_paths_span, &value_span);
            disallowed_path.set_span(span_from_toml_range($file, value_span));
            disallowed_paths.push(disallowed_path);
        }
        (disallowed_paths, disallowed_paths_span)
    }};
}

macro_rules! define_Conf {
    ($(
        $(#[doc = $doc:literal])+
        $(#[conf_deprecated($dep:literal, $new_conf:ident)])?
        $(#[default_text = $default_text:expr])?
        $(#[disallowed_paths_allow_replacements = $replacements_allowed:expr])?
        $(#[lints($($for_lints:ident),* $(,)?)])?
        $name:ident: $ty:ty = $default:expr,
    )*) => {
        /// Clippy lint configuration
        pub struct Conf {
            $($(#[cfg_attr(doc, doc = $doc)])+ pub $name: $ty,)*
        }

        mod defaults {
            use super::*;
            $(pub fn $name() -> $ty { $default })*
        }

        impl Default for Conf {
            fn default() -> Self {
                Self { $($name: defaults::$name(),)* }
            }
        }

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "kebab-case")]
        #[allow(non_camel_case_types)]
        enum Field { $($name,)* third_party, }

        struct ConfVisitor<'a>(&'a SourceFile);

        impl<'de> Visitor<'de> for ConfVisitor<'_> {
            type Value = TryConf;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("Conf")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error> where V: MapAccess<'de> {
                let mut value_spans = HashMap::new();
                let mut errors = Vec::new();
                let mut warnings = Vec::new();

                // Declare a local variable for each field available to a configuration file.
                $(let mut $name = None;)*

                // could get `Field` here directly, but get `String` first for diagnostics
                while let Some(name) = map.next_key::<toml::Spanned<String>>()? {
                    let field = match Field::deserialize(name.get_ref().as_str().into_deserializer()) {
                        Err(e) => {
                            let e: FieldError = e;
                            errors.push(ConfError::spanned(self.0, e.error, e.suggestion, name.span()));
                            continue;
                        }
                        Ok(field) => field
                    };

                    match field {
                        $(Field::$name => {
                            // Is this a deprecated field, i.e., is `$dep` set? If so, push a warning.
                            $(warnings.push(ConfError::spanned(self.0, format!("deprecated field `{}`. {}", name.get_ref(), $dep), None, name.span()));)?
                            let (value, value_span) =
                                deserialize!(map, $ty, errors, self.0 $(, $replacements_allowed)?);
                            // Was this field set previously?
                            if $name.is_some() {
                                errors.push(ConfError::spanned(self.0, format!("duplicate field `{}`", name.get_ref()), None, name.span()));
                                continue;
                            }
                            $name = Some(value);
                            value_spans.insert(name.get_ref().as_str().to_string(), value_span);
                            // If this is a deprecated field, was the new field (`$new_conf`) set previously?
                            // Note that `$new_conf` is one of the defined `$name`s.
                            $(match $new_conf {
                                Some(_) => errors.push(ConfError::spanned(self.0, concat!(
                                    "duplicate field `", stringify!($new_conf),
                                    "` (provided as `", stringify!($name), "`)"
                                ), None, name.span())),
                                None => $new_conf = $name.clone(),
                            })?
                        })*
                        // ignore contents of the third_party key
                        Field::third_party => drop(map.next_value::<IgnoredAny>())
                    }
                }
                let conf = Conf { $($name: $name.unwrap_or_else(defaults::$name),)* };
                Ok(TryConf { conf, value_spans, errors, warnings })
            }
        }

        pub fn get_configuration_metadata() -> Vec<ClippyConfiguration> {
            vec![$(
                ClippyConfiguration {
                    name: stringify!($name).replace('_', "-"),
                    default: default_text!(defaults::$name() $(, $default_text)?),
                    lints: &[$($(stringify!($for_lints)),*)?],
                    doc: concat!($($doc, '\n',)*),
                    deprecation_reason: wrap_option!($($dep)?)
                },
            )*]
        }
    };
}

fn union(x: &Range<usize>, y: &Range<usize>) -> Range<usize> {
    Range {
        start: cmp::min(x.start, y.start),
        end: cmp::max(x.end, y.end),
    }
}

fn span_from_toml_range(file: &SourceFile, span: Range<usize>) -> Span {
    Span::new(
        file.start_pos + BytePos::from_usize(span.start),
        file.start_pos + BytePos::from_usize(span.end),
        SyntaxContext::root(),
        None,
    )
}

define_Conf! {
    /// Which crates to allow absolute paths from
    #[lints(absolute_paths)]
    absolute_paths_allowed_crates: Vec<String> = Vec::new(),
    /// The maximum number of segments a path can have before being linted, anything above this will
    /// be linted.
    #[lints(absolute_paths)]
    absolute_paths_max_segments: u64 = 2,
    /// Whether to accept a safety comment to be placed above the attributes for the `unsafe` block
    #[lints(undocumented_unsafe_blocks)]
    accept_comment_above_attributes: bool = true,
    /// Whether to accept a safety comment to be placed above the statement containing the `unsafe` block
    #[lints(undocumented_unsafe_blocks)]
    accept_comment_above_statement: bool = true,
    /// Don't lint when comparing the result of a modulo operation to zero.
    #[lints(modulo_arithmetic)]
    allow_comparison_to_zero: bool = true,
    /// Whether `dbg!` should be allowed in test functions or `#[cfg(test)]`
    #[lints(dbg_macro)]
    allow_dbg_in_tests: bool = false,
    /// Whether an item should be allowed to have the same name as its containing module
    #[lints(module_name_repetitions)]
    allow_exact_repetitions: bool = true,
    /// Whether `expect` should be allowed in code always evaluated at compile time
    #[lints(expect_used)]
    allow_expect_in_consts: bool = true,
    /// Whether `expect` should be allowed in test functions or `#[cfg(test)]`
    #[lints(expect_used)]
    allow_expect_in_tests: bool = false,
    /// Whether `indexing_slicing` should be allowed in test functions or `#[cfg(test)]`
    #[lints(indexing_slicing)]
    allow_indexing_slicing_in_tests: bool = false,
    /// Whether to allow mixed uninlined format args, e.g. `format!("{} {}", a, foo.bar)`
    #[lints(uninlined_format_args)]
    allow_mixed_uninlined_format_args: bool = true,
    /// Whether to allow `r#""#` when `r""` can be used
    #[lints(needless_raw_string_hashes)]
    allow_one_hash_in_raw_strings: bool = false,
    /// Whether `panic` should be allowed in test functions or `#[cfg(test)]`
    #[lints(panic)]
    allow_panic_in_tests: bool = false,
    /// Whether print macros (ex. `println!`) should be allowed in test functions or `#[cfg(test)]`
    #[lints(print_stderr, print_stdout)]
    allow_print_in_tests: bool = false,
    /// Whether to allow module inception if it's not public.
    #[lints(module_inception)]
    allow_private_module_inception: bool = false,
    /// List of trait paths to ignore when checking renamed function parameters.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allow-renamed-params-for = [ "std::convert::From" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// - By default, the following traits are ignored: `From`, `TryFrom`, `FromStr`
    /// - `".."` can be used as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(renamed_function_params)]
    allow_renamed_params_for: Vec<String> =
        DEFAULT_ALLOWED_TRAITS_WITH_RENAMED_PARAMS.iter().map(ToString::to_string).collect(),
    /// Whether `unwrap` should be allowed in code always evaluated at compile time
    #[lints(unwrap_used)]
    allow_unwrap_in_consts: bool = true,
    /// Whether `unwrap` should be allowed in test functions or `#[cfg(test)]`
    #[lints(unwrap_used)]
    allow_unwrap_in_tests: bool = false,
    /// Whether `useless_vec` should ignore test functions or `#[cfg(test)]`
    #[lints(useless_vec)]
    allow_useless_vec_in_tests: bool = false,
    /// Additional dotfiles (files or directories starting with a dot) to allow
    #[lints(path_ends_with_ext)]
    allowed_dotfiles: Vec<String> = Vec::default(),
    /// A list of crate names to allow duplicates of
    #[lints(multiple_crate_versions)]
    allowed_duplicate_crates: Vec<String> = Vec::new(),
    /// Allowed names below the minimum allowed characters. The value `".."` can be used as part of
    /// the list to indicate, that the configured values should be appended to the default
    /// configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(min_ident_chars)]
    allowed_idents_below_min_chars: Vec<String> =
        DEFAULT_ALLOWED_IDENTS_BELOW_MIN_CHARS.iter().map(ToString::to_string).collect(),
    /// List of prefixes to allow when determining whether an item's name ends with the module's name.
    /// If the rest of an item's name is an allowed prefix (e.g. item `ToFoo` or `to_foo` in module `foo`),
    /// then don't emit a warning.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allowed-prefixes = [ "to", "from" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// - By default, the following prefixes are allowed: `to`, `as`, `into`, `from`, `try_into` and `try_from`
    /// - PascalCase variant is included automatically for each snake_case variant (e.g. if `try_into` is included,
    ///   `TryInto` will also be included)
    /// - Use `".."` as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value
    #[lints(module_name_repetitions)]
    allowed_prefixes: Vec<String> = DEFAULT_ALLOWED_PREFIXES.iter().map(ToString::to_string).collect(),
    /// The list of unicode scripts allowed to be used in the scope.
    #[lints(disallowed_script_idents)]
    allowed_scripts: Vec<String> = vec!["Latin".to_string()],
    /// List of path segments allowed to have wildcard imports.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allowed-wildcard-imports = [ "utils", "common" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// 1. This configuration has no effects if used with `warn_on_all_wildcard_imports = true`.
    /// 2. Paths with any segment that containing the word 'prelude'
    /// are already allowed by default.
    #[lints(wildcard_imports)]
    allowed_wildcard_imports: Vec<String> = Vec::new(),
    /// Suppress checking of the passed type names in all types of operations.
    ///
    /// If a specific operation is desired, consider using `arithmetic_side_effects_allowed_binary` or `arithmetic_side_effects_allowed_unary` instead.
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed = ["SomeType", "AnotherType"]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// A type, say `SomeType`, listed in this configuration has the same behavior of
    /// `["SomeType" , "*"], ["*", "SomeType"]` in `arithmetic_side_effects_allowed_binary`.
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed: Vec<String> = <_>::default(),
    /// Suppress checking of the passed type pair names in binary operations like addition or
    /// multiplication.
    ///
    /// Supports the "*" wildcard to indicate that a certain type won't trigger the lint regardless
    /// of the involved counterpart. For example, `["SomeType", "*"]` or `["*", "AnotherType"]`.
    ///
    /// Pairs are asymmetric, which means that `["SomeType", "AnotherType"]` is not the same as
    /// `["AnotherType", "SomeType"]`.
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed-binary = [["SomeType" , "f32"], ["AnotherType", "*"]]
    /// ```
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed_binary: Vec<(String, String)> = <_>::default(),
    /// Suppress checking of the passed type names in unary operations like "negation" (`-`).
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed-unary = ["SomeType", "AnotherType"]
    /// ```
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed_unary: Vec<String> = <_>::default(),
    /// The maximum allowed size for arrays on the stack
    #[lints(large_const_arrays, large_stack_arrays)]
    array_size_threshold: u64 = 16 * 1024,
    /// Suppress lints whenever the suggested change would cause breakage for other crates.
    #[lints(
        box_collection,
        enum_variant_names,
        large_types_passed_by_value,
        linkedlist,
        needless_pass_by_ref_mut,
        option_option,
        owned_cow,
        rc_buffer,
        rc_mutex,
        redundant_allocation,
        ref_option,
        single_call_fn,
        trivially_copy_pass_by_ref,
        unnecessary_box_returns,
        unnecessary_wraps,
        unused_self,
        upper_case_acronyms,
        vec_box,
        wrong_self_convention,
    )]
    avoid_breaking_exported_api: bool = true,
    /// The list of types which may not be held across an await point.
    #[disallowed_paths_allow_replacements = false]
    #[lints(await_holding_invalid_type)]
    await_holding_invalid_types: Vec<DisallowedPathWithoutReplacement> = Vec::new(),
    /// DEPRECATED LINT: BLACKLISTED_NAME.
    ///
    /// Use the Disallowed Names lint instead
    #[conf_deprecated("Please use `disallowed-names` instead", disallowed_names)]
    blacklisted_names: Vec<String> = Vec::new(),
    /// For internal testing only, ignores the current `publish` settings in the Cargo manifest.
    #[lints(cargo_common_metadata)]
    cargo_ignore_publish: bool = false,
    /// Whether to check MSRV compatibility in `#[test]` and `#[cfg(test)]` code.
    #[lints(incompatible_msrv)]
    check_incompatible_msrv_in_tests: bool = false,
    /// Whether to suggest reordering constructor fields when initializers are present.
    ///
    /// Warnings produced by this configuration aren't necessarily fixed by just reordering the fields. Even if the
    /// suggested code would compile, it can change semantics if the initializer expressions have side effects. The
    /// following example [from rust-clippy#11846] shows how the suggestion can run into borrow check errors:
    ///
    /// ```rust
    /// struct MyStruct {
    ///     vector: Vec<u32>,
    ///     length: usize
    /// }
    /// fn main() {
    ///     let vector = vec![1,2,3];
    ///     MyStruct { length: vector.len(), vector};
    /// }
    /// ```
    ///
    /// [from rust-clippy#11846]: https://github.com/rust-lang/rust-clippy/issues/11846#issuecomment-1820747924
    #[lints(inconsistent_struct_constructor)]
    check_inconsistent_struct_field_initializers: bool = false,
    /// Whether to also run the listed lints on private items.
    #[lints(missing_errors_doc, missing_panics_doc, missing_safety_doc, unnecessary_safety_doc)]
    check_private_items: bool = false,
    /// The maximum cognitive complexity a function can have
    #[lints(cognitive_complexity)]
    cognitive_complexity_threshold: u64 = 25,
    /// The minimum digits a const float literal must have to supress the `excessive_precicion` lint
    #[lints(excessive_precision)]
    const_literal_digits_threshold: usize = 30,
    /// DEPRECATED LINT: CYCLOMATIC_COMPLEXITY.
    ///
    /// Use the Cognitive Complexity lint instead.
    #[conf_deprecated("Please use `cognitive-complexity-threshold` instead", cognitive_complexity_threshold)]
    cyclomatic_complexity_threshold: u64 = 25,
    /// The list of disallowed macros, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the macro that should be disallowed
    /// - `reason` (optional): explanation why this macro is disallowed
    /// - `replacement` (optional): suggested alternative macro
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[disallowed_paths_allow_replacements = true]
    #[lints(disallowed_macros)]
    disallowed_macros: Vec<DisallowedPath> = Vec::new(),
    /// The list of disallowed methods, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the method that should be disallowed
    /// - `reason` (optional): explanation why this method is disallowed
    /// - `replacement` (optional): suggested alternative method
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[disallowed_paths_allow_replacements = true]
    #[lints(disallowed_methods)]
    disallowed_methods: Vec<DisallowedPath> = Vec::new(),
    /// The list of disallowed names to lint about. NB: `bar` is not here since it has legitimate uses. The value
    /// `".."` can be used as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(disallowed_names)]
    disallowed_names: Vec<String> = DEFAULT_DISALLOWED_NAMES.iter().map(ToString::to_string).collect(),
    /// The list of disallowed types, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the type that should be disallowed
    /// - `reason` (optional): explanation why this type is disallowed
    /// - `replacement` (optional): suggested alternative type
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[disallowed_paths_allow_replacements = true]
    #[lints(disallowed_types)]
    disallowed_types: Vec<DisallowedPath> = Vec::new(),
    /// The list of words this lint should not consider as identifiers needing ticks. The value
    /// `".."` can be used as part of the list to indicate, that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value. For example:
    /// * `doc-valid-idents = ["ClipPy"]` would replace the default list with `["ClipPy"]`.
    /// * `doc-valid-idents = ["ClipPy", ".."]` would append `ClipPy` to the default list.
    #[lints(doc_markdown)]
    doc_valid_idents: Vec<String> = DEFAULT_DOC_VALID_IDENTS.iter().map(ToString::to_string).collect(),
    /// Whether to apply the raw pointer heuristic to determine if a type is `Send`.
    #[lints(non_send_fields_in_send_ty)]
    enable_raw_pointer_heuristic_for_send: bool = true,
    /// Whether to recommend using implicit into iter for reborrowed values.
    ///
    /// #### Example
    /// ```no_run
    /// let mut vec = vec![1, 2, 3];
    /// let rmvec = &mut vec;
    /// for _ in rmvec.iter() {}
    /// for _ in rmvec.iter_mut() {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let mut vec = vec![1, 2, 3];
    /// let rmvec = &mut vec;
    /// for _ in &*rmvec {}
    /// for _ in &mut *rmvec {}
    /// ```
    #[lints(explicit_iter_loop)]
    enforce_iter_loop_reborrow: bool = false,
    /// The list of imports to always rename, a fully qualified path followed by the rename.
    #[lints(missing_enforced_import_renames)]
    enforced_import_renames: Vec<Rename> = Vec::new(),
    /// The minimum number of enum variants for the lints about variant names to trigger
    #[lints(enum_variant_names)]
    enum_variant_name_threshold: u64 = 3,
    /// The maximum size of an enum's variant to avoid box suggestion
    #[lints(large_enum_variant)]
    enum_variant_size_threshold: u64 = 200,
    /// The maximum amount of nesting a block can reside in
    #[lints(excessive_nesting)]
    excessive_nesting_threshold: u64 = 0,
    /// The maximum byte size a `Future` can have, before it triggers the `clippy::large_futures` lint
    #[lints(large_futures)]
    future_size_threshold: u64 = 16 * 1024,
    /// A list of paths to types that should be treated as if they do not contain interior mutability
    #[lints(borrow_interior_mutable_const, declare_interior_mutable_const, ifs_same_cond, mutable_key_type)]
    ignore_interior_mutability: Vec<String> = Vec::from(["bytes::Bytes".into()]),
    /// The maximum size of the `Err`-variant in a `Result` returned from a function
    #[lints(result_large_err)]
    large_error_threshold: u64 = 128,
    /// Whether collapsible `if` and `else if` chains are linted if they contain comments inside the parts
    /// that would be collapsed.
    #[lints(collapsible_else_if, collapsible_if)]
    lint_commented_code: bool = false,
    /// Whether to suggest reordering constructor fields when initializers are present.
    /// DEPRECATED CONFIGURATION: lint-inconsistent-struct-field-initializers
    ///
    /// Use the `check-inconsistent-struct-field-initializers` configuration instead.
    #[conf_deprecated("Please use `check-inconsistent-struct-field-initializers` instead", check_inconsistent_struct_field_initializers)]
    lint_inconsistent_struct_field_initializers: bool = false,
    /// The lower bound for linting decimal literals
    #[lints(decimal_literal_representation)]
    literal_representation_threshold: u64 = 16384,
    /// Whether the matches should be considered by the lint, and whether there should
    /// be filtering for common types.
    #[lints(manual_let_else)]
    matches_for_let_else: MatchLintBehaviour = MatchLintBehaviour::WellKnownTypes,
    /// The maximum number of bool parameters a function can have
    #[lints(fn_params_excessive_bools)]
    max_fn_params_bools: u64 = 3,
    /// The maximum size of a file included via `include_bytes!()` or `include_str!()`, in bytes
    #[lints(large_include_file)]
    max_include_file_size: u64 = 1_000_000,
    /// The maximum number of bool fields a struct can have
    #[lints(struct_excessive_bools)]
    max_struct_bools: u64 = 3,
    /// When Clippy suggests using a slice pattern, this is the maximum number of elements allowed in
    /// the slice pattern that is suggested. If more elements are necessary, the lint is suppressed.
    /// For example, `[_, _, _, e, ..]` is a slice pattern with 4 elements.
    #[lints(index_refutable_slice)]
    max_suggested_slice_pattern_length: u64 = 3,
    /// The maximum number of bounds a trait can have to be linted
    #[lints(type_repetition_in_bounds)]
    max_trait_bounds: u64 = 3,
    /// Minimum chars an ident can have, anything below or equal to this will be linted.
    #[lints(min_ident_chars)]
    min_ident_chars_threshold: u64 = 1,
    /// Whether to allow fields starting with an underscore to skip documentation requirements
    #[lints(missing_docs_in_private_items)]
    missing_docs_allow_unused: bool = false,
    /// Whether to **only** check for missing documentation in items visible within the current
    /// crate. For example, `pub(crate)` items.
    #[lints(missing_docs_in_private_items)]
    missing_docs_in_crate_items: bool = false,
    /// The named groupings of different source item kinds within modules.
    #[lints(arbitrary_source_item_ordering)]
    module_item_order_groupings: SourceItemOrderingModuleItemGroupings = DEFAULT_MODULE_ITEM_ORDERING_GROUPS.into(),
    /// Whether the items within module groups should be ordered alphabetically or not.
    ///
    /// This option can be configured to "all", "none", or a list of specific grouping names that should be checked
    /// (e.g. only "enums").
    #[lints(arbitrary_source_item_ordering)]
    module_items_ordered_within_groupings: SourceItemOrderingWithinModuleItemGroupings =
        SourceItemOrderingWithinModuleItemGroupings::None,
    /// The minimum rust version that the project supports. Defaults to the `rust-version` field in `Cargo.toml`
    #[default_text = "current version"]
    #[lints(
        allow_attributes,
        allow_attributes_without_reason,
        almost_complete_range,
        approx_constant,
        assigning_clones,
        borrow_as_ptr,
        cast_abs_to_unsigned,
        checked_conversions,
        cloned_instead_of_copied,
        collapsible_match,
        collapsible_str_replace,
        deprecated_cfg_attr,
        derivable_impls,
        err_expect,
        filter_map_next,
        from_over_into,
        if_then_some_else_none,
        index_refutable_slice,
        io_other_error,
        iter_kv_map,
        legacy_numeric_constants,
        lines_filter_map_ok,
        manual_abs_diff,
        manual_bits,
        manual_c_str_literals,
        manual_clamp,
        manual_div_ceil,
        manual_flatten,
        manual_hash_one,
        manual_is_ascii_check,
        manual_is_power_of_two,
        manual_let_else,
        manual_midpoint,
        manual_non_exhaustive,
        manual_option_as_slice,
        manual_pattern_char_comparison,
        manual_range_contains,
        manual_rem_euclid,
        manual_repeat_n,
        manual_retain,
        manual_slice_fill,
        manual_slice_size_calculation,
        manual_split_once,
        manual_str_repeat,
        manual_strip,
        manual_try_fold,
        map_clone,
        map_unwrap_or,
        map_with_unused_argument_over_ranges,
        match_like_matches_macro,
        mem_replace_option_with_some,
        mem_replace_with_default,
        missing_const_for_fn,
        needless_borrow,
        non_std_lazy_statics,
        option_as_ref_deref,
        ptr_as_ptr,
        question_mark,
        redundant_field_names,
        redundant_static_lifetimes,
        repeat_vec_with_capacity,
        same_item_push,
        seek_from_current,
        to_digit_is_some,
        transmute_ptr_to_ref,
        tuple_array_conversions,
        type_repetition_in_bounds,
        unchecked_duration_subtraction,
        uninlined_format_args,
        unnecessary_lazy_evaluations,
        unnested_or_patterns,
        unused_trait_names,
        use_self,
        zero_ptr,
    )]
    msrv: Msrv = Msrv::default(),
    /// The minimum size (in bytes) to consider a type for passing by reference instead of by value.
    #[lints(large_types_passed_by_value)]
    pass_by_value_size_limit: u64 = 256,
    /// Lint "public" fields in a struct that are prefixed with an underscore based on their
    /// exported visibility, or whether they are marked as "pub".
    #[lints(pub_underscore_fields)]
    pub_underscore_fields_behavior: PubUnderscoreFieldsBehaviour = PubUnderscoreFieldsBehaviour::PubliclyExported,
    /// Whether to lint only if it's multiline.
    #[lints(semicolon_inside_block)]
    semicolon_inside_block_ignore_singleline: bool = false,
    /// Whether to lint only if it's singleline.
    #[lints(semicolon_outside_block)]
    semicolon_outside_block_ignore_multiline: bool = false,
    /// The maximum number of single char bindings a scope may have
    #[lints(many_single_char_names)]
    single_char_binding_names_threshold: u64 = 4,
    /// Which kind of elements should be ordered internally, possible values being `enum`, `impl`, `module`, `struct`, `trait`.
    #[lints(arbitrary_source_item_ordering)]
    source_item_ordering: SourceItemOrdering = DEFAULT_SOURCE_ITEM_ORDERING.into(),
    /// The maximum allowed stack size for functions in bytes
    #[lints(large_stack_frames)]
    stack_size_threshold: u64 = 512_000,
    /// Enforce the named macros always use the braces specified.
    ///
    /// A `MacroMatcher` can be added like so `{ name = "macro_name", brace = "(" }`. If the macro
    /// could be used with a full path two `MacroMatcher`s have to be added one with the full path
    /// `crate_name::macro_name` and one with just the macro name.
    #[lints(nonstandard_macro_braces)]
    standard_macro_braces: Vec<MacroMatcher> = Vec::new(),
    /// The minimum number of struct fields for the lints about field names to trigger
    #[lints(struct_field_names)]
    struct_field_name_threshold: u64 = 3,
    /// Whether to suppress a restriction lint in constant code. In same
    /// cases the restructured operation might not be unavoidable, as the
    /// suggested counterparts are unavailable in constant code. This
    /// configuration will cause restriction lints to trigger even
    /// if no suggestion can be made.
    #[lints(indexing_slicing)]
    suppress_restriction_lint_in_const: bool = false,
    /// The maximum size of objects (in bytes) that will be linted. Larger objects are ok on the heap
    #[lints(boxed_local, useless_vec)]
    too_large_for_stack: u64 = 200,
    /// The maximum number of argument a function or method can have
    #[lints(too_many_arguments)]
    too_many_arguments_threshold: u64 = 7,
    /// The maximum number of lines a function or method can have
    #[lints(too_many_lines)]
    too_many_lines_threshold: u64 = 100,
    /// The order of associated items in traits.
    #[lints(arbitrary_source_item_ordering)]
    trait_assoc_item_kinds_order: SourceItemOrderingTraitAssocItemKinds = DEFAULT_TRAIT_ASSOC_ITEM_KINDS_ORDER.into(),
    /// The maximum size (in bytes) to consider a `Copy` type for passing by value instead of by
    /// reference.
    #[default_text = "target_pointer_width"]
    #[lints(trivially_copy_pass_by_ref)]
    trivial_copy_size_limit: Option<u64> = None,
    /// The maximum complexity a type can have
    #[lints(type_complexity)]
    type_complexity_threshold: u64 = 250,
    /// The byte size a `T` in `Box<T>` can have, below which it triggers the `clippy::unnecessary_box` lint
    #[lints(unnecessary_box_returns)]
    unnecessary_box_size: u64 = 128,
    /// Should the fraction of a decimal be linted to include separators.
    #[lints(unreadable_literal)]
    unreadable_literal_lint_fractions: bool = true,
    /// Enables verbose mode. Triggers if there is more than one uppercase char next to each other
    #[lints(upper_case_acronyms)]
    upper_case_acronyms_aggressive: bool = false,
    /// The size of the boxed type in bytes, where boxing in a `Vec` is allowed
    #[lints(vec_box)]
    vec_box_size_threshold: u64 = 4096,
    /// The maximum allowed size of a bit mask before suggesting to use 'trailing_zeros'
    #[lints(verbose_bit_mask)]
    verbose_bit_mask_threshold: u64 = 1,
    /// Whether to emit warnings on all wildcard imports, including those from `prelude`, from `super` in tests,
    /// or for `pub use` reexports.
    #[lints(wildcard_imports)]
    warn_on_all_wildcard_imports: bool = false,
    /// Whether to also emit warnings for unsafe blocks with metavariable expansions in **private** macros.
    #[lints(macro_metavars_in_unsafe)]
    warn_unsafe_macro_metavars_in_private_macros: bool = false,
}

/// Search for the configuration file.
///
/// # Errors
///
/// Returns any unexpected filesystem error encountered when searching for the config file
pub fn lookup_conf_file() -> io::Result<(Option<PathBuf>, Vec<String>)> {
    /// Possible filename to search for.
    const CONFIG_FILE_NAMES: [&str; 2] = [".clippy.toml", "clippy.toml"];

    // Start looking for a config file in CLIPPY_CONF_DIR, or failing that, CARGO_MANIFEST_DIR.
    // If neither of those exist, use ".". (Update documentation if this priority changes)
    let mut current = env::var_os("CLIPPY_CONF_DIR")
        .or_else(|| env::var_os("CARGO_MANIFEST_DIR"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .canonicalize()?;

    let mut found_config: Option<PathBuf> = None;
    let mut warnings = vec![];

    loop {
        for config_file_name in &CONFIG_FILE_NAMES {
            if let Ok(config_file) = current.join(config_file_name).canonicalize() {
                match fs::metadata(&config_file) {
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {},
                    Err(e) => return Err(e),
                    Ok(md) if md.is_dir() => {},
                    Ok(_) => {
                        // warn if we happen to find two config files #8323
                        if let Some(ref found_config) = found_config {
                            warnings.push(format!(
                                "using config file `{}`, `{}` will be ignored",
                                found_config.display(),
                                config_file.display()
                            ));
                        } else {
                            found_config = Some(config_file);
                        }
                    },
                }
            }
        }

        if found_config.is_some() {
            return Ok((found_config, warnings));
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Ok((None, warnings));
        }
    }
}

fn deserialize(file: &SourceFile) -> TryConf {
    match toml::de::Deserializer::new(file.src.as_ref().unwrap()).deserialize_map(ConfVisitor(file)) {
        Ok(mut conf) => {
            extend_vec_if_indicator_present(&mut conf.conf.disallowed_names, DEFAULT_DISALLOWED_NAMES);
            extend_vec_if_indicator_present(&mut conf.conf.allowed_prefixes, DEFAULT_ALLOWED_PREFIXES);
            extend_vec_if_indicator_present(
                &mut conf.conf.allow_renamed_params_for,
                DEFAULT_ALLOWED_TRAITS_WITH_RENAMED_PARAMS,
            );

            // Confirms that the user has not accidentally configured ordering requirements for groups that
            // aren't configured.
            if let SourceItemOrderingWithinModuleItemGroupings::Custom(groupings) =
                &conf.conf.module_items_ordered_within_groupings
            {
                for grouping in groupings {
                    if !conf.conf.module_item_order_groupings.is_grouping(grouping) {
                        // Since this isn't fixable by rustfix, don't emit a `Suggestion`. This just adds some useful
                        // info for the user instead.

                        let names = conf.conf.module_item_order_groupings.grouping_names();
                        let suggestion = suggest_candidate(grouping, names.iter().map(String::as_str))
                            .map(|s| format!(" perhaps you meant `{s}`?"))
                            .unwrap_or_default();
                        let names = names.iter().map(|s| format!("`{s}`")).join(", ");
                        let message = format!(
                            "unknown ordering group: `{grouping}` was not specified in `module-items-ordered-within-groupings`,{suggestion} expected one of: {names}"
                        );

                        let span = conf
                            .value_spans
                            .get("module_item_order_groupings")
                            .cloned()
                            .unwrap_or_default();
                        conf.errors.push(ConfError::spanned(file, message, None, span));
                    }
                }
            }

            // TODO: THIS SHOULD BE TESTED, this comment will be gone soon
            if conf.conf.allowed_idents_below_min_chars.iter().any(|e| e == "..") {
                conf.conf
                    .allowed_idents_below_min_chars
                    .extend(DEFAULT_ALLOWED_IDENTS_BELOW_MIN_CHARS.iter().map(ToString::to_string));
            }
            if conf.conf.doc_valid_idents.iter().any(|e| e == "..") {
                conf.conf
                    .doc_valid_idents
                    .extend(DEFAULT_DOC_VALID_IDENTS.iter().map(ToString::to_string));
            }

            conf
        },
        Err(e) => TryConf::from_toml_error(file, &e),
    }
}

fn extend_vec_if_indicator_present(vec: &mut Vec<String>, default: &[&str]) {
    if vec.contains(&"..".to_string()) {
        vec.extend(default.iter().map(ToString::to_string));
    }
}

impl Conf {
    pub fn read(sess: &Session, path: &io::Result<(Option<PathBuf>, Vec<String>)>) -> &'static Conf {
        static CONF: OnceLock<Conf> = OnceLock::new();
        CONF.get_or_init(|| Conf::read_inner(sess, path))
    }

    fn read_inner(sess: &Session, path: &io::Result<(Option<PathBuf>, Vec<String>)>) -> Conf {
        match path {
            Ok((_, warnings)) => {
                for warning in warnings {
                    sess.dcx().warn(warning.clone());
                }
            },
            Err(error) => {
                sess.dcx()
                    .err(format!("error finding Clippy's configuration file: {error}"));
            },
        }

        let TryConf {
            mut conf,
            value_spans: _,
            errors,
            warnings,
        } = match path {
            Ok((Some(path), _)) => match sess.source_map().load_file(path) {
                Ok(file) => deserialize(&file),
                Err(error) => {
                    sess.dcx().err(format!("failed to read `{}`: {error}", path.display()));
                    TryConf::default()
                },
            },
            _ => TryConf::default(),
        };

        conf.msrv.read_cargo(sess);

        // all conf errors are non-fatal, we just use the default conf in case of error
        for error in errors {
            let mut diag = sess.dcx().struct_span_err(
                error.span,
                format!("error reading Clippy's configuration file: {}", error.message),
            );

            if let Some(sugg) = error.suggestion {
                diag.span_suggestion(error.span, sugg.message, sugg.suggestion, Applicability::MaybeIncorrect);
            }

            diag.emit();
        }

        for warning in warnings {
            sess.dcx().span_warn(
                warning.span,
                format!("error reading Clippy's configuration file: {}", warning.message),
            );
        }

        conf
    }
}

const SEPARATOR_WIDTH: usize = 4;

#[derive(Debug)]
struct FieldError {
    error: String,
    suggestion: Option<Suggestion>,
}

#[derive(Debug)]
struct Suggestion {
    message: &'static str,
    suggestion: &'static str,
}

impl std::error::Error for FieldError {}

impl Display for FieldError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.pad(&self.error)
    }
}

impl serde::de::Error for FieldError {
    fn custom<T: Display>(msg: T) -> Self {
        Self {
            error: msg.to_string(),
            suggestion: None,
        }
    }

    fn unknown_field(field: &str, expected: &'static [&'static str]) -> Self {
        // List the available fields sorted and at least one per line, more if `CLIPPY_TERMINAL_WIDTH` is
        // set and allows it.
        use fmt::Write;

        let metadata = get_configuration_metadata();
        let deprecated = metadata
            .iter()
            .filter_map(|conf| {
                if conf.deprecation_reason.is_some() {
                    Some(conf.name.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut expected = expected
            .iter()
            .copied()
            .filter(|name| !deprecated.contains(name))
            .collect::<Vec<_>>();
        expected.sort_unstable();

        let (rows, column_widths) = calculate_dimensions(&expected);

        let mut msg = format!("unknown field `{field}`, expected one of");
        for row in 0..rows {
            writeln!(msg).unwrap();
            for (column, column_width) in column_widths.iter().copied().enumerate() {
                let index = column * rows + row;
                let field = expected.get(index).copied().unwrap_or_default();
                write!(msg, "{:SEPARATOR_WIDTH$}{field:column_width$}", " ").unwrap();
            }
        }

        let suggestion = suggest_candidate(field, expected).map(|suggestion| Suggestion {
            message: "perhaps you meant",
            suggestion,
        });

        Self { error: msg, suggestion }
    }
}

fn calculate_dimensions(fields: &[&str]) -> (usize, Vec<usize>) {
    let columns = env::var("CLIPPY_TERMINAL_WIDTH")
        .ok()
        .and_then(|s| <usize as FromStr>::from_str(&s).ok())
        .map_or(1, |terminal_width| {
            let max_field_width = fields.iter().map(|field| field.len()).max().unwrap();
            cmp::max(1, terminal_width / (SEPARATOR_WIDTH + max_field_width))
        });

    let rows = fields.len().div_ceil(columns);

    let column_widths = (0..columns)
        .map(|column| {
            if column < columns - 1 {
                (0..rows)
                    .map(|row| {
                        let index = column * rows + row;
                        let field = fields.get(index).copied().unwrap_or_default();
                        field.len()
                    })
                    .max()
                    .unwrap()
            } else {
                // Avoid adding extra space to the last column.
                0
            }
        })
        .collect::<Vec<_>>();

    (rows, column_widths)
}

/// Given a user-provided value that couldn't be matched to a known option, finds the most likely
/// candidate among candidates that the user might have meant.
fn suggest_candidate<'a, I>(value: &str, candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    candidates
        .into_iter()
        .filter_map(|expected| {
            let dist = edit_distance(value, expected, 4)?;
            Some((dist, expected))
        })
        .min_by_key(|&(dist, _)| dist)
        .map(|(_, suggestion)| suggestion)
}

#[cfg(test)]
mod tests {
    use serde::de::IgnoredAny;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use walkdir::WalkDir;

    #[test]
    fn configs_are_tested() {
        let mut names: HashSet<String> = crate::get_configuration_metadata()
            .into_iter()
            .filter_map(|meta| {
                if meta.deprecation_reason.is_none() {
                    Some(meta.name.replace('_', "-"))
                } else {
                    None
                }
            })
            .collect();

        let toml_files = WalkDir::new("../tests")
            .into_iter()
            .map(Result::unwrap)
            .filter(|entry| entry.file_name() == "clippy.toml");

        for entry in toml_files {
            let file = fs::read_to_string(entry.path()).unwrap();
            #[allow(clippy::zero_sized_map_values)]
            if let Ok(map) = toml::from_str::<HashMap<String, IgnoredAny>>(&file) {
                for name in map.keys() {
                    names.remove(name.as_str());
                }
            }
        }

        assert!(
            names.is_empty(),
            "Configuration variable lacks test: {names:?}\nAdd a test to `tests/ui-toml`"
        );
    }
}
