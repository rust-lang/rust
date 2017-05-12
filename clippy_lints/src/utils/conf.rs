//! Read configurations files.

#![deny(missing_docs_in_private_items)]

use std::{env, fmt, fs, io, path};
use std::io::Read;
use syntax::{ast, codemap};
use toml;
use std::sync::Mutex;

/// Get the configuration file from arguments.
pub fn file_from_args(args: &[codemap::Spanned<ast::NestedMetaItemKind>])
    -> Result<Option<path::PathBuf>, (&'static str, codemap::Span)> {
    for arg in args.iter().filter_map(|a| a.meta_item()) {
        if arg.name() == "conf_file" {
            return match arg.node {
                ast::MetaItemKind::Word |
                ast::MetaItemKind::List(_) => Err(("`conf_file` must be a named value", arg.span)),
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
    /// Type error.
    Type(/// The name of the key.
         &'static str,
         /// The expected type.
         &'static str,
         /// The type we got instead.
         &'static str),
    /// There is an unknown key is the file.
    UnknownKey(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::Toml(ref err) => err.fmt(f),
            Error::Type(key, expected, got) => {
                write!(f, "`{}` is expected to be a `{}` but is a `{}`", key, expected, got)
            },
            Error::UnknownKey(ref key) => write!(f, "unknown key `{}`", key),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

lazy_static! {
    static ref ERRORS: Mutex<Vec<Error>> = Mutex::new(Vec::new());
}

macro_rules! define_Conf {
    ($(#[$doc: meta] ($rust_name: ident, $rust_name_str: expr, $default: expr => $($ty: tt)+),)+) => {
        pub use self::helpers::Conf;
        mod helpers {
            /// Type used to store lint configuration.
            #[derive(Deserialize)]
            #[serde(rename_all="kebab-case")]
            #[serde(deny_unknown_fields)]
            pub struct Conf {
                $(#[$doc] #[serde(default=$rust_name_str)] #[serde(with=$rust_name_str)] pub $rust_name: define_Conf!(TY $($ty)+),)+
                #[allow(dead_code)]
                #[serde(default)]
                third_party: Option<::toml::Value>,
            }
            $(
                mod $rust_name {
                    use serde;
                    use serde::Deserialize;
                    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<define_Conf!(TY $($ty)+), D::Error> {
                        type T = define_Conf!(TY $($ty)+);
                        Ok(T::deserialize(deserializer).unwrap_or_else(|e| {
                            ::utils::conf::ERRORS.lock().expect("no threading here").push(::utils::conf::Error::Toml(e.to_string()));
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

    // how to read the value?
    (CONV i64, $value: expr) => { $value.as_integer() };
    (CONV u64, $value: expr) => {
        $value.as_integer()
        .iter()
        .filter_map(|&i| if i >= 0 { Some(i as u64) } else { None })
        .next()
    };
    (CONV String, $value: expr) => { $value.as_str().map(Into::into) };
    (CONV Vec<String>, $value: expr) => {{
        let slice = $value.as_array();

        if let Some(slice) = slice {
            if slice.iter().any(|v| v.as_str().is_none()) {
                None
            } else {
                Some(slice.iter().map(|v| v.as_str().expect("already checked").to_owned()).collect())
            }
        } else {
            None
        }
    }};

    // provide a nicer syntax to declare the default value of `Vec<String>` variables
    (DEFAULT Vec<String>, $e: expr) => { $e.iter().map(|&e| e.to_owned()).collect() };
    (DEFAULT $ty: ty, $e: expr) => { $e };
}

define_Conf! {
    /// Lint: BLACKLISTED_NAME. The list of blacklisted names to lint about
    (blacklisted_names, "blacklisted_names", ["foo", "bar", "baz", "quux"] => Vec<String>),
    /// Lint: CYCLOMATIC_COMPLEXITY. The maximum cyclomatic complexity a function can have
    (cyclomatic_complexity_threshold, "cyclomatic_complexity_threshold", 25 => u64),
    /// Lint: DOC_MARKDOWN. The list of words this lint should not consider as identifiers needing ticks
    (doc_valid_idents, "doc_valid_idents", [
        "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
        "DirectX",
        "ECMAScript",
        "GPLv2", "GPLv3",
        "GitHub",
        "IPv4", "IPv6",
        "JavaScript",
        "NaN",
        "OAuth",
        "OpenGL",
        "TrueType",
        "iOS", "macOS",
        "TeX", "LaTeX", "BibTex", "BibLaTex",
        "MinGW",
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
    /// Lint: LARGE_ENUM_VARIANT. The maximum size of a emum's variant to avoid box suggestion
    (enum_variant_size_threshold, "enum_variant_size_threshold", 200 => u64),
}

/// Search for the configuration file.
pub fn lookup_conf_file() -> io::Result<Option<path::PathBuf>> {
    /// Possible filename to search for.
    const CONFIG_FILE_NAMES: [&'static str; 2] = [".clippy.toml", "clippy.toml"];

    let mut current = try!(env::current_dir());

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
    (toml::from_str("").expect("we never error on empty config files"), errors)
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
        Ok(toml) => (toml, ERRORS.lock().expect("no threading -> mutex always safe").split_off(0)),
        Err(e) => {
            let mut errors = ERRORS.lock().expect("no threading -> mutex always safe").split_off(0);
            errors.push(Error::Toml(e.to_string()));
            default(errors)
        },
    }
}
