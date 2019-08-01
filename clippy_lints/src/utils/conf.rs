//! Read configurations files.

#![deny(clippy::missing_docs_in_private_items)]

use lazy_static::lazy_static;
use std::default::Default;
use std::io::Read;
use std::sync::Mutex;
use std::{env, fmt, fs, io, path};
use syntax::{ast, source_map};
use toml;

/// Gets the configuration file from arguments.
pub fn file_from_args(args: &[ast::NestedMetaItem]) -> Result<Option<path::PathBuf>, (&'static str, source_map::Span)> {
    for arg in args.iter().filter_map(syntax::ast::NestedMetaItem::meta_item) {
        if arg.check_name(sym!(conf_file)) {
            return match arg.node {
                ast::MetaItemKind::Word | ast::MetaItemKind::List(_) => {
                    Err(("`conf_file` must be a named value", arg.span))
                },
                ast::MetaItemKind::NameValue(ref value) => {
                    if let ast::LitKind::Str(ref file, _) = value.node {
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
    /// Not valid toml or doesn't fit the expected conf format
    Toml(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Self::Io(ref err) => err.fmt(f),
            Self::Toml(ref err) => err.fmt(f),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

lazy_static! {
    static ref ERRORS: Mutex<Vec<Error>> = Mutex::new(Vec::new());
}

macro_rules! define_Conf {
    ($(#[$doc: meta] ($rust_name: ident, $rust_name_str: expr, $default: expr => $($ty: tt)+),)+) => {
        pub use self::helpers::Conf;
        mod helpers {
            use serde::Deserialize;
            /// Type used to store lint configuration.
            #[derive(Deserialize)]
            #[serde(rename_all="kebab-case", deny_unknown_fields)]
            pub struct Conf {
                $(#[$doc] #[serde(default=$rust_name_str)] #[serde(with=$rust_name_str)]
                          pub $rust_name: define_Conf!(TY $($ty)+),)+
                #[allow(dead_code)]
                #[serde(default)]
                third_party: Option<::toml::Value>,
            }
            $(
                mod $rust_name {
                    use serde;
                    use serde::Deserialize;
                    crate fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D)
                    -> Result<define_Conf!(TY $($ty)+), D::Error> {
                        type T = define_Conf!(TY $($ty)+);
                        Ok(T::deserialize(deserializer).unwrap_or_else(|e| {
                            crate::utils::conf::ERRORS.lock().expect("no threading here")
                                                        .push(crate::utils::conf::Error::Toml(e.to_string()));
                            super::$rust_name()
                        }))
                    }
                }

                fn $rust_name() -> define_Conf!(TY $($ty)+) {
                    define_Conf!(DEFAULT $($ty)+, $default)
                }
            )+
        }
    };

    // hack to convert tts
    (TY $ty: ty) => { $ty };

    // provide a nicer syntax to declare the default value of `Vec<String>` variables
    (DEFAULT Vec<String>, $e: expr) => { $e.iter().map(|&e| e.to_owned()).collect() };
    (DEFAULT $ty: ty, $e: expr) => { $e };
}

define_Conf! {
    /// Lint: BLACKLISTED_NAME. The list of blacklisted names to lint about
    (blacklisted_names, "blacklisted_names", ["foo", "bar", "baz", "quux"] => Vec<String>),
    /// Lint: COGNITIVE_COMPLEXITY. The maximum cognitive complexity a function can have
    (cognitive_complexity_threshold, "cognitive_complexity_threshold", 25 => u64),
    /// DEPRECATED LINT: CYCLOMATIC_COMPLEXITY. Use the Cognitive Complexity lint instead.
    (cyclomatic_complexity_threshold, "cyclomatic_complexity_threshold", None => Option<u64>),
    /// Lint: DOC_MARKDOWN. The list of words this lint should not consider as identifiers needing ticks
    (doc_valid_idents, "doc_valid_idents", [
        "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
        "DirectX",
        "ECMAScript",
        "GPLv2", "GPLv3",
        "GitHub", "GitLab",
        "IPv4", "IPv6",
        "JavaScript",
        "NaN", "NaNs",
        "OAuth",
        "OpenGL", "OpenSSH", "OpenSSL", "OpenStreetMap",
        "TrueType",
        "iOS", "macOS",
        "TeX", "LaTeX", "BibTeX", "BibLaTeX",
        "MinGW",
        "CamelCase",
    ] => Vec<String>),
    /// Lint: TOO_MANY_ARGUMENTS. The maximum number of argument a function or method can have
    (too_many_arguments_threshold, "too_many_arguments_threshold", 7 => u64),
    /// Lint: TYPE_COMPLEXITY. The maximum complexity a type can have
    (type_complexity_threshold, "type_complexity_threshold", 250 => u64),
    /// Lint: MANY_SINGLE_CHAR_NAMES. The maximum number of single char bindings a scope may have
    (single_char_binding_names_threshold, "single_char_binding_names_threshold", 5 => u64),
    /// Lint: BOXED_LOCAL. The maximum size of objects (in bytes) that will be linted. Larger objects are ok on the heap
    (too_large_for_stack, "too_large_for_stack", 200 => u64),
    /// Lint: ENUM_VARIANT_NAMES. The minimum number of enum variants for the lints about variant names to trigger
    (enum_variant_name_threshold, "enum_variant_name_threshold", 3 => u64),
    /// Lint: LARGE_ENUM_VARIANT. The maximum size of a enum's variant to avoid box suggestion
    (enum_variant_size_threshold, "enum_variant_size_threshold", 200 => u64),
    /// Lint: VERBOSE_BIT_MASK. The maximum allowed size of a bit mask before suggesting to use 'trailing_zeros'
    (verbose_bit_mask_threshold, "verbose_bit_mask_threshold", 1 => u64),
    /// Lint: DECIMAL_LITERAL_REPRESENTATION. The lower bound for linting decimal literals
    (literal_representation_threshold, "literal_representation_threshold", 16384 => u64),
    /// Lint: TRIVIALLY_COPY_PASS_BY_REF. The maximum size (in bytes) to consider a `Copy` type for passing by value instead of by reference.
    (trivial_copy_size_limit, "trivial_copy_size_limit", None => Option<u64>),
    /// Lint: TOO_MANY_LINES. The maximum number of lines a function or method can have
    (too_many_lines_threshold, "too_many_lines_threshold", 100 => u64),
}

impl Default for Conf {
    fn default() -> Self {
        toml::from_str("").expect("we never error on empty config files")
    }
}

/// Search for the configuration file.
pub fn lookup_conf_file() -> io::Result<Option<path::PathBuf>> {
    /// Possible filename to search for.
    const CONFIG_FILE_NAMES: [&str; 2] = [".clippy.toml", "clippy.toml"];

    // Start looking for a config file in CLIPPY_CONF_DIR, or failing that, CARGO_MANIFEST_DIR.
    // If neither of those exist, use ".".
    let mut current = path::PathBuf::from(
        env::var("CLIPPY_CONF_DIR")
            .or_else(|_| env::var("CARGO_MANIFEST_DIR"))
            .unwrap_or_else(|_| ".".to_string()),
    );
    loop {
        for config_file_name in &CONFIG_FILE_NAMES {
            let config_file = current.join(config_file_name);
            match fs::metadata(&config_file) {
                // Only return if it's a file to handle the unlikely situation of a directory named
                // `clippy.toml`.
                Ok(ref md) if md.is_file() => return Ok(Some(config_file)),
                // Return the error if it's something other than `NotFound`; otherwise we didn't
                // find the project file yet, and continue searching.
                Err(e) => {
                    if e.kind() != io::ErrorKind::NotFound {
                        return Err(e);
                    }
                },
                _ => (),
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
pub fn read(path: Option<&path::Path>) -> (Conf, Vec<Error>) {
    let path = if let Some(path) = path {
        path
    } else {
        return default(Vec::new());
    };

    let file = match fs::File::open(path) {
        Ok(mut file) => {
            let mut buf = String::new();

            if let Err(err) = file.read_to_string(&mut buf) {
                return default(vec![err.into()]);
            }

            buf
        },
        Err(err) => return default(vec![err.into()]),
    };

    assert!(ERRORS.lock().expect("no threading -> mutex always safe").is_empty());
    match toml::from_str(&file) {
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
