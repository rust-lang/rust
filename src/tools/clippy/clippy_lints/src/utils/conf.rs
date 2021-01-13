//! Read configurations files.

#![deny(clippy::missing_docs_in_private_items)]

use rustc_ast::ast::{LitKind, MetaItemKind, NestedMetaItem};
use rustc_span::source_map;
use source_map::Span;
use std::lazy::SyncLazy;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::{env, fmt, fs, io};

/// Gets the configuration file from arguments.
pub fn file_from_args(args: &[NestedMetaItem]) -> Result<Option<PathBuf>, (&'static str, Span)> {
    for arg in args.iter().filter_map(NestedMetaItem::meta_item) {
        if arg.has_name(sym!(conf_file)) {
            return match arg.kind {
                MetaItemKind::Word | MetaItemKind::List(_) => Err(("`conf_file` must be a named value", arg.span)),
                MetaItemKind::NameValue(ref value) => {
                    if let LitKind::Str(ref file, _) = value.kind {
                        Ok(Some(file.to_string().into()))
                    } else {
                        Err(("`conf_file` value must be a string", value.span))
                    }
                },
            };
        }
    }

    Ok(None)
}

/// Error from reading a configuration file.
#[derive(Debug)]
pub enum Error {
    /// An I/O error.
    Io(io::Error),
    /// Not valid toml or doesn't fit the expected config format
    Toml(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => err.fmt(f),
            Self::Toml(err) => err.fmt(f),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Vec of errors that might be collected during config toml parsing
static ERRORS: SyncLazy<Mutex<Vec<Error>>> = SyncLazy::new(|| Mutex::new(Vec::new()));

macro_rules! define_Conf {
    ($(#[$doc:meta] ($config:ident, $config_str:literal: $Ty:ty, $default:expr),)+) => {
        mod helpers {
            use serde::Deserialize;
            /// Type used to store lint configuration.
            #[derive(Deserialize)]
            #[serde(rename_all = "kebab-case", deny_unknown_fields)]
            pub struct Conf {
                $(
                    #[$doc]
                    #[serde(default = $config_str)]
                    #[serde(with = $config_str)]
                    pub $config: $Ty,
                )+
                #[allow(dead_code)]
                #[serde(default)]
                third_party: Option<::toml::Value>,
            }

            $(
                mod $config {
                    use serde::Deserialize;
                    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<$Ty, D::Error> {
                        use super::super::{ERRORS, Error};

                        Ok(
                            <$Ty>::deserialize(deserializer).unwrap_or_else(|e| {
                                ERRORS
                                    .lock()
                                    .expect("no threading here")
                                    .push(Error::Toml(e.to_string()));
                                super::$config()
                            })
                        )
                    }
                }

                #[must_use]
                fn $config() -> $Ty {
                    let x = $default;
                    x
                }
            )+
        }
    };
}

pub use self::helpers::Conf;
define_Conf! {
    /// Lint: REDUNDANT_FIELD_NAMES, REDUNDANT_STATIC_LIFETIMES, FILTER_MAP_NEXT, CHECKED_CONVERSIONS, MANUAL_RANGE_CONTAINS, USE_SELF, MEM_REPLACE_WITH_DEFAULT, MANUAL_NON_EXHAUSTIVE, OPTION_AS_REF_DEREF, MAP_UNWRAP_OR, MATCH_LIKE_MATCHES_MACRO, MANUAL_STRIP, MISSING_CONST_FOR_FN. The minimum rust version that the project supports
    (msrv, "msrv": Option<String>, None),
    /// Lint: BLACKLISTED_NAME. The list of blacklisted names to lint about. NB: `bar` is not here since it has legitimate uses
    (blacklisted_names, "blacklisted_names": Vec<String>, ["foo", "baz", "quux"].iter().map(ToString::to_string).collect()),
    /// Lint: COGNITIVE_COMPLEXITY. The maximum cognitive complexity a function can have
    (cognitive_complexity_threshold, "cognitive_complexity_threshold": u64, 25),
    /// DEPRECATED LINT: CYCLOMATIC_COMPLEXITY. Use the Cognitive Complexity lint instead.
    (cyclomatic_complexity_threshold, "cyclomatic_complexity_threshold": Option<u64>, None),
    /// Lint: DOC_MARKDOWN. The list of words this lint should not consider as identifiers needing ticks
    (doc_valid_idents, "doc_valid_idents": Vec<String>, [
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
        "OpenGL", "OpenMP", "OpenSSH", "OpenSSL", "OpenStreetMap",
        "TensorFlow",
        "TrueType",
        "iOS", "macOS",
        "TeX", "LaTeX", "BibTeX", "BibLaTeX",
        "MinGW",
        "CamelCase",
    ].iter().map(ToString::to_string).collect()),
    /// Lint: TOO_MANY_ARGUMENTS. The maximum number of argument a function or method can have
    (too_many_arguments_threshold, "too_many_arguments_threshold": u64, 7),
    /// Lint: TYPE_COMPLEXITY. The maximum complexity a type can have
    (type_complexity_threshold, "type_complexity_threshold": u64, 250),
    /// Lint: MANY_SINGLE_CHAR_NAMES. The maximum number of single char bindings a scope may have
    (single_char_binding_names_threshold, "single_char_binding_names_threshold": u64, 4),
    /// Lint: BOXED_LOCAL, USELESS_VEC. The maximum size of objects (in bytes) that will be linted. Larger objects are ok on the heap
    (too_large_for_stack, "too_large_for_stack": u64, 200),
    /// Lint: ENUM_VARIANT_NAMES. The minimum number of enum variants for the lints about variant names to trigger
    (enum_variant_name_threshold, "enum_variant_name_threshold": u64, 3),
    /// Lint: LARGE_ENUM_VARIANT. The maximum size of a enum's variant to avoid box suggestion
    (enum_variant_size_threshold, "enum_variant_size_threshold": u64, 200),
    /// Lint: VERBOSE_BIT_MASK. The maximum allowed size of a bit mask before suggesting to use 'trailing_zeros'
    (verbose_bit_mask_threshold, "verbose_bit_mask_threshold": u64, 1),
    /// Lint: DECIMAL_LITERAL_REPRESENTATION. The lower bound for linting decimal literals
    (literal_representation_threshold, "literal_representation_threshold": u64, 16384),
    /// Lint: TRIVIALLY_COPY_PASS_BY_REF. The maximum size (in bytes) to consider a `Copy` type for passing by value instead of by reference.
    (trivial_copy_size_limit, "trivial_copy_size_limit": Option<u64>, None),
    /// Lint: LARGE_TYPE_PASS_BY_MOVE. The minimum size (in bytes) to consider a type for passing by reference instead of by value.
    (pass_by_value_size_limit, "pass_by_value_size_limit": u64, 256),
    /// Lint: TOO_MANY_LINES. The maximum number of lines a function or method can have
    (too_many_lines_threshold, "too_many_lines_threshold": u64, 100),
    /// Lint: LARGE_STACK_ARRAYS, LARGE_CONST_ARRAYS. The maximum allowed size for arrays on the stack
    (array_size_threshold, "array_size_threshold": u64, 512_000),
    /// Lint: VEC_BOX. The size of the boxed type in bytes, where boxing in a `Vec` is allowed
    (vec_box_size_threshold, "vec_box_size_threshold": u64, 4096),
    /// Lint: TYPE_REPETITION_IN_BOUNDS. The maximum number of bounds a trait can have to be linted
    (max_trait_bounds, "max_trait_bounds": u64, 3),
    /// Lint: STRUCT_EXCESSIVE_BOOLS. The maximum number of bools a struct can have
    (max_struct_bools, "max_struct_bools": u64, 3),
    /// Lint: FN_PARAMS_EXCESSIVE_BOOLS. The maximum number of bools function parameters can have
    (max_fn_params_bools, "max_fn_params_bools": u64, 3),
    /// Lint: WILDCARD_IMPORTS. Whether to allow certain wildcard imports (prelude, super in tests).
    (warn_on_all_wildcard_imports, "warn_on_all_wildcard_imports": bool, false),
    /// Lint: DISALLOWED_METHOD. The list of blacklisted methods to lint about. NB: `bar` is not here since it has legitimate uses
    (disallowed_methods, "disallowed_methods": Vec<String>, Vec::<String>::new()),
    /// Lint: UNREADABLE_LITERAL. Should the fraction of a decimal be linted to include separators.
    (unreadable_literal_lint_fractions, "unreadable_literal_lint_fractions": bool, true),
}

impl Default for Conf {
    #[must_use]
    fn default() -> Self {
        toml::from_str("").expect("we never error on empty config files")
    }
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
            let config_file = current.join(config_file_name);
            match fs::metadata(&config_file) {
                // Only return if it's a file to handle the unlikely situation of a directory named
                // `clippy.toml`.
                Ok(ref md) if !md.is_dir() => return Ok(Some(config_file)),
                // Return the error if it's something other than `NotFound`; otherwise we didn't
                // find the project file yet, and continue searching.
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                _ => {},
            }
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Ok(None);
        }
    }
}

/// Produces a `Conf` filled with the default values and forwards the errors
///
/// Used internally for convenience
fn default(errors: Vec<Error>) -> (Conf, Vec<Error>) {
    (Conf::default(), errors)
}

/// Read the `toml` configuration file.
///
/// In case of error, the function tries to continue as much as possible.
pub fn read(path: &Path) -> (Conf, Vec<Error>) {
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => return default(vec![err.into()]),
    };

    assert!(ERRORS.lock().expect("no threading -> mutex always safe").is_empty());
    match toml::from_str(&content) {
        Ok(toml) => {
            let mut errors = ERRORS.lock().expect("no threading -> mutex always safe").split_off(0);

            let toml_ref: &Conf = &toml;

            let cyc_field: Option<u64> = toml_ref.cyclomatic_complexity_threshold;

            if cyc_field.is_some() {
                let cyc_err = "found deprecated field `cyclomatic-complexity-threshold`. Please use `cognitive-complexity-threshold` instead.".to_string();
                errors.push(Error::Toml(cyc_err));
            }

            (toml, errors)
        },
        Err(e) => {
            let mut errors = ERRORS.lock().expect("no threading -> mutex always safe").split_off(0);
            errors.push(Error::Toml(e.to_string()));

            default(errors)
        },
    }
}
