//! Read configurations files.

#![allow(clippy::module_name_repetitions)]

use serde::de::{Deserializer, IgnoredAny, IntoDeserializer, MapAccess, Visitor};
use serde::Deserialize;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::{env, fmt, fs, io};

/// Holds information used by `MISSING_ENFORCED_IMPORT_RENAMES` lint.
#[derive(Clone, Debug, Deserialize)]
pub struct Rename {
    pub path: String,
    pub rename: String,
}

/// A single disallowed method, used by the `DISALLOWED_METHOD` lint.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum DisallowedMethod {
    Simple(String),
    WithReason { path: String, reason: Option<String> },
}

/// A single disallowed type, used by the `DISALLOWED_TYPE` lint.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum DisallowedType {
    Simple(String),
    WithReason { path: String, reason: Option<String> },
}

/// Conf with parse errors
#[derive(Default)]
pub struct TryConf {
    pub conf: Conf,
    pub errors: Vec<String>,
}

impl TryConf {
    fn from_error(error: impl Error) -> Self {
        Self {
            conf: Conf::default(),
            errors: vec![error.to_string()],
        }
    }
}

macro_rules! define_Conf {
    ($(
        $(#[doc = $doc:literal])+
        $(#[conf_deprecated($dep:literal)])?
        ($name:ident: $ty:ty = $default:expr),
    )*) => {
        /// Clippy lint configuration
        pub struct Conf {
            $($(#[doc = $doc])+ pub $name: $ty,)*
        }

        mod defaults {
            $(pub fn $name() -> $ty { $default })*
        }

        impl Default for Conf {
            fn default() -> Self {
                Self { $($name: defaults::$name(),)* }
            }
        }

        impl<'de> Deserialize<'de> for TryConf {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                deserializer.deserialize_map(ConfVisitor)
            }
        }

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "kebab-case")]
        #[allow(non_camel_case_types)]
        enum Field { $($name,)* third_party, }

        struct ConfVisitor;

        impl<'de> Visitor<'de> for ConfVisitor {
            type Value = TryConf;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("Conf")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error> where V: MapAccess<'de> {
                let mut errors = Vec::new();
                $(let mut $name = None;)*
                // could get `Field` here directly, but get `str` first for diagnostics
                while let Some(name) = map.next_key::<&str>()? {
                    match Field::deserialize(name.into_deserializer())? {
                        $(Field::$name => {
                            $(errors.push(format!("deprecated field `{}`. {}", name, $dep));)?
                            match map.next_value() {
                                Err(e) => errors.push(e.to_string()),
                                Ok(value) => match $name {
                                    Some(_) => errors.push(format!("duplicate field `{}`", name)),
                                    None => $name = Some(value),
                                }
                            }
                        })*
                        // white-listed; ignore
                        Field::third_party => drop(map.next_value::<IgnoredAny>())
                    }
                }
                let conf = Conf { $($name: $name.unwrap_or_else(defaults::$name),)* };
                Ok(TryConf { conf, errors })
            }
        }

        #[cfg(feature = "metadata-collector-lint")]
        pub mod metadata {
            use crate::utils::internal_lints::metadata_collector::ClippyConfiguration;

            macro_rules! wrap_option {
                () => (None);
                ($x:literal) => (Some($x));
            }

            pub(crate) fn get_configuration_metadata() -> Vec<ClippyConfiguration> {
                vec![
                    $(
                        {
                            let deprecation_reason = wrap_option!($($dep)?);

                            ClippyConfiguration::new(
                                stringify!($name),
                                stringify!($ty),
                                format!("{:?}", super::defaults::$name()),
                                concat!($($doc, '\n',)*),
                                deprecation_reason,
                            )
                        },
                    )+
                ]
            }
        }
    };
}

define_Conf! {
    /// Lint: ENUM_VARIANT_NAMES, LARGE_TYPES_PASSED_BY_VALUE, TRIVIALLY_COPY_PASS_BY_REF, UNNECESSARY_WRAPS, UPPER_CASE_ACRONYMS, WRONG_SELF_CONVENTION, BOX_COLLECTION, REDUNDANT_ALLOCATION, RC_BUFFER, VEC_BOX, OPTION_OPTION, LINKEDLIST, RC_MUTEX.
    ///
    /// Suppress lints whenever the suggested change would cause breakage for other crates.
    (avoid_breaking_exported_api: bool = true),
    /// Lint: MANUAL_SPLIT_ONCE, MANUAL_STR_REPEAT, CLONED_INSTEAD_OF_COPIED, REDUNDANT_FIELD_NAMES, REDUNDANT_STATIC_LIFETIMES, FILTER_MAP_NEXT, CHECKED_CONVERSIONS, MANUAL_RANGE_CONTAINS, USE_SELF, MEM_REPLACE_WITH_DEFAULT, MANUAL_NON_EXHAUSTIVE, OPTION_AS_REF_DEREF, MAP_UNWRAP_OR, MATCH_LIKE_MATCHES_MACRO, MANUAL_STRIP, MISSING_CONST_FOR_FN, UNNESTED_OR_PATTERNS, FROM_OVER_INTO, PTR_AS_PTR, IF_THEN_SOME_ELSE_NONE, APPROX_CONSTANT.
    ///
    /// The minimum rust version that the project supports
    (msrv: Option<String> = None),
    /// Lint: BLACKLISTED_NAME.
    ///
    /// The list of blacklisted names to lint about. NB: `bar` is not here since it has legitimate uses
    (blacklisted_names: Vec<String> = ["foo", "baz", "quux"].iter().map(ToString::to_string).collect()),
    /// Lint: COGNITIVE_COMPLEXITY.
    ///
    /// The maximum cognitive complexity a function can have
    (cognitive_complexity_threshold: u64 = 25),
    /// DEPRECATED LINT: CYCLOMATIC_COMPLEXITY.
    ///
    /// Use the Cognitive Complexity lint instead.
    #[conf_deprecated("Please use `cognitive-complexity-threshold` instead")]
    (cyclomatic_complexity_threshold: Option<u64> = None),
    /// Lint: DOC_MARKDOWN.
    ///
    /// The list of words this lint should not consider as identifiers needing ticks
    (doc_valid_idents: Vec<String> = [
        "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
        "DirectX",
        "ECMAScript",
        "GPLv2", "GPLv3",
        "GitHub", "GitLab",
        "IPv4", "IPv6",
        "ClojureScript", "CoffeeScript", "JavaScript", "PureScript", "TypeScript",
        "NaN", "NaNs",
        "OAuth", "GraphQL",
        "OCaml",
        "OpenGL", "OpenMP", "OpenSSH", "OpenSSL", "OpenStreetMap", "OpenDNS",
        "WebGL",
        "TensorFlow",
        "TrueType",
        "iOS", "macOS", "FreeBSD",
        "TeX", "LaTeX", "BibTeX", "BibLaTeX",
        "MinGW",
        "CamelCase",
    ].iter().map(ToString::to_string).collect()),
    /// Lint: TOO_MANY_ARGUMENTS.
    ///
    /// The maximum number of argument a function or method can have
    (too_many_arguments_threshold: u64 = 7),
    /// Lint: TYPE_COMPLEXITY.
    ///
    /// The maximum complexity a type can have
    (type_complexity_threshold: u64 = 250),
    /// Lint: MANY_SINGLE_CHAR_NAMES.
    ///
    /// The maximum number of single char bindings a scope may have
    (single_char_binding_names_threshold: u64 = 4),
    /// Lint: BOXED_LOCAL, USELESS_VEC.
    ///
    /// The maximum size of objects (in bytes) that will be linted. Larger objects are ok on the heap
    (too_large_for_stack: u64 = 200),
    /// Lint: ENUM_VARIANT_NAMES.
    ///
    /// The minimum number of enum variants for the lints about variant names to trigger
    (enum_variant_name_threshold: u64 = 3),
    /// Lint: LARGE_ENUM_VARIANT.
    ///
    /// The maximum size of an enum's variant to avoid box suggestion
    (enum_variant_size_threshold: u64 = 200),
    /// Lint: VERBOSE_BIT_MASK.
    ///
    /// The maximum allowed size of a bit mask before suggesting to use 'trailing_zeros'
    (verbose_bit_mask_threshold: u64 = 1),
    /// Lint: DECIMAL_LITERAL_REPRESENTATION.
    ///
    /// The lower bound for linting decimal literals
    (literal_representation_threshold: u64 = 16384),
    /// Lint: TRIVIALLY_COPY_PASS_BY_REF.
    ///
    /// The maximum size (in bytes) to consider a `Copy` type for passing by value instead of by reference.
    (trivial_copy_size_limit: Option<u64> = None),
    /// Lint: LARGE_TYPE_PASS_BY_MOVE.
    ///
    /// The minimum size (in bytes) to consider a type for passing by reference instead of by value.
    (pass_by_value_size_limit: u64 = 256),
    /// Lint: TOO_MANY_LINES.
    ///
    /// The maximum number of lines a function or method can have
    (too_many_lines_threshold: u64 = 100),
    /// Lint: LARGE_STACK_ARRAYS, LARGE_CONST_ARRAYS.
    ///
    /// The maximum allowed size for arrays on the stack
    (array_size_threshold: u64 = 512_000),
    /// Lint: VEC_BOX.
    ///
    /// The size of the boxed type in bytes, where boxing in a `Vec` is allowed
    (vec_box_size_threshold: u64 = 4096),
    /// Lint: TYPE_REPETITION_IN_BOUNDS.
    ///
    /// The maximum number of bounds a trait can have to be linted
    (max_trait_bounds: u64 = 3),
    /// Lint: STRUCT_EXCESSIVE_BOOLS.
    ///
    /// The maximum number of bool fields a struct can have
    (max_struct_bools: u64 = 3),
    /// Lint: FN_PARAMS_EXCESSIVE_BOOLS.
    ///
    /// The maximum number of bool parameters a function can have
    (max_fn_params_bools: u64 = 3),
    /// Lint: WILDCARD_IMPORTS.
    ///
    /// Whether to allow certain wildcard imports (prelude, super in tests).
    (warn_on_all_wildcard_imports: bool = false),
    /// Lint: DISALLOWED_METHOD.
    ///
    /// The list of disallowed methods, written as fully qualified paths.
    (disallowed_methods: Vec<crate::utils::conf::DisallowedMethod> = Vec::new()),
    /// Lint: DISALLOWED_TYPE.
    ///
    /// The list of disallowed types, written as fully qualified paths.
    (disallowed_types: Vec<crate::utils::conf::DisallowedType> = Vec::new()),
    /// Lint: UNREADABLE_LITERAL.
    ///
    /// Should the fraction of a decimal be linted to include separators.
    (unreadable_literal_lint_fractions: bool = true),
    /// Lint: UPPER_CASE_ACRONYMS.
    ///
    /// Enables verbose mode. Triggers if there is more than one uppercase char next to each other
    (upper_case_acronyms_aggressive: bool = false),
    /// Lint: _CARGO_COMMON_METADATA.
    ///
    /// For internal testing only, ignores the current `publish` settings in the Cargo manifest.
    (cargo_ignore_publish: bool = false),
    /// Lint: NONSTANDARD_MACRO_BRACES.
    ///
    /// Enforce the named macros always use the braces specified.
    ///
    /// A `MacroMatcher` can be added like so `{ name = "macro_name", brace = "(" }`. If the macro
    /// is could be used with a full path two `MacroMatcher`s have to be added one with the full path
    /// `crate_name::macro_name` and one with just the macro name.
    (standard_macro_braces: Vec<crate::nonstandard_macro_braces::MacroMatcher> = Vec::new()),
    /// Lint: MISSING_ENFORCED_IMPORT_RENAMES.
    ///
    /// The list of imports to always rename, a fully qualified path followed by the rename.
    (enforced_import_renames: Vec<crate::utils::conf::Rename> = Vec::new()),
    /// Lint: RESTRICTED_SCRIPTS.
    ///
    /// The list of unicode scripts allowed to be used in the scope.
    (allowed_scripts: Vec<String> = vec!["Latin".to_string()]),
    /// Lint: NON_SEND_FIELDS_IN_SEND_TY.
    ///
    /// Whether to apply the raw pointer heuristic to determine if a type is `Send`.
    (enable_raw_pointer_heuristic_for_send: bool = true),
}

/// Search for the configuration file.
pub fn lookup_conf_file() -> io::Result<Option<PathBuf>> {
    /// Possible filename to search for.
    const CONFIG_FILE_NAMES: [&str; 2] = [".clippy.toml", "clippy.toml"];

    // Start looking for a config file in CLIPPY_CONF_DIR, or failing that, CARGO_MANIFEST_DIR.
    // If neither of those exist, use ".".
    let mut current = env::var_os("CLIPPY_CONF_DIR")
        .or_else(|| env::var_os("CARGO_MANIFEST_DIR"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    loop {
        for config_file_name in &CONFIG_FILE_NAMES {
            if let Ok(config_file) = current.join(config_file_name).canonicalize() {
                match fs::metadata(&config_file) {
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {},
                    Err(e) => return Err(e),
                    Ok(md) if md.is_dir() => {},
                    Ok(_) => return Ok(Some(config_file)),
                }
            }
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Ok(None);
        }
    }
}

/// Read the `toml` configuration file.
///
/// In case of error, the function tries to continue as much as possible.
pub fn read(path: &Path) -> TryConf {
    let content = match fs::read_to_string(path) {
        Err(e) => return TryConf::from_error(e),
        Ok(content) => content,
    };
    toml::from_str(&content).unwrap_or_else(TryConf::from_error)
}
