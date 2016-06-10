use std::{fmt, fs, io};
use std::io::Read;
use syntax::{ast, codemap, ptr};
use syntax::parse::token;
use toml;

/// Get the configuration file from arguments.
pub fn file(args: &[ptr::P<ast::MetaItem>]) -> Result<Option<token::InternedString>, (&'static str, codemap::Span)> {
    for arg in args {
        match arg.node {
            ast::MetaItemKind::Word(ref name) |
            ast::MetaItemKind::List(ref name, _) => {
                if name == &"conf_file" {
                    return Err(("`conf_file` must be a named value", arg.span));
                }
            }
            ast::MetaItemKind::NameValue(ref name, ref value) => {
                if name == &"conf_file" {
                    return if let ast::LitKind::Str(ref file, _) = value.node {
                        Ok(Some(file.clone()))
                    } else {
                        Err(("`conf_file` value must be a string", value.span))
                    };
                }
            }
        }
    }

    Ok(None)
}

/// Error from reading a configuration file.
#[derive(Debug)]
pub enum Error {
    IoError(io::Error),
    TomlError(Vec<toml::ParserError>),
    TypeError(&'static str, &'static str, &'static str),
    UnknownKey(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Error::IoError(ref err) => err.fmt(f),
            Error::TomlError(ref errs) => {
                let mut first = true;
                for err in errs {
                    if !first {
                        try!(", ".fmt(f));
                        first = false;
                    }

                    try!(err.fmt(f));
                }

                Ok(())
            }
            Error::TypeError(ref key, ref expected, ref got) => {
                write!(f, "`{}` is expected to be a `{}` but is a `{}`", key, expected, got)
            }
            Error::UnknownKey(ref key) => write!(f, "unknown key `{}`", key),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IoError(e)
    }
}

macro_rules! define_Conf {
    ($(#[$doc: meta] ($toml_name: tt, $rust_name: ident, $default: expr => $($ty: tt)+),)+) => {
        /// Type used to store lint configuration.
        pub struct Conf {
            $(#[$doc] pub $rust_name: define_Conf!(TY $($ty)+),)+
        }

        impl Default for Conf {
            fn default() -> Conf {
                Conf {
                    $($rust_name: define_Conf!(DEFAULT $($ty)+, $default),)+
                }
            }
        }

        impl Conf {
            /// Set the property `name` (which must be the `toml` name) to the given value
            #[allow(cast_sign_loss)]
            fn set(&mut self, name: String, value: toml::Value) -> Result<(), Error> {
                match name.as_str() {
                    $(
                        define_Conf!(PAT $toml_name) => {
                            if let Some(value) = define_Conf!(CONV $($ty)+, value) {
                                self.$rust_name = value;
                            }
                            else {
                                return Err(Error::TypeError(define_Conf!(EXPR $toml_name),
                                                                stringify!($($ty)+),
                                                                value.type_str()));
                            }
                        },
                    )+
                    "third-party" => {
                        // for external tools such as clippy-service
                        return Ok(());
                    }
                    _ => {
                        return Err(Error::UnknownKey(name));
                    }
                }

                Ok(())
            }
        }
    };

    // hack to convert tts
    (PAT $pat: pat) => { $pat };
    (EXPR $e: expr) => { $e };
    (TY $ty: ty) => { $ty };

    // how to read the value?
    (CONV i64, $value: expr) => { $value.as_integer() };
    (CONV u64, $value: expr) => { $value.as_integer().iter().filter_map(|&i| if i >= 0 { Some(i as u64) } else { None }).next() };
    (CONV String, $value: expr) => { $value.as_str().map(Into::into) };
    (CONV Vec<String>, $value: expr) => {{
        let slice = $value.as_slice();

        if let Some(slice) = slice {
            if slice.iter().any(|v| v.as_str().is_none()) {
                None
            }
            else {
                Some(slice.iter().map(|v| v.as_str().unwrap_or_else(|| unreachable!()).to_owned()).collect())
            }
        }
        else {
            None
        }
    }};

    // provide a nicer syntax to declare the default value of `Vec<String>` variables
    (DEFAULT Vec<String>, $e: expr) => { $e.iter().map(|&e| e.to_owned()).collect() };
    (DEFAULT $ty: ty, $e: expr) => { $e };
}

define_Conf! {
    /// Lint: BLACKLISTED_NAME. The list of blacklisted names to lint about
    ("blacklisted-names", blacklisted_names, ["foo", "bar", "baz"] => Vec<String>),
    /// Lint: CYCLOMATIC_COMPLEXITY. The maximum cyclomatic complexity a function can have
    ("cyclomatic-complexity-threshold", cyclomatic_complexity_threshold, 25 => u64),
    /// Lint: DOC_MARKDOWN. The list of words this lint should not consider as identifiers needing ticks
    ("doc-valid-idents", doc_valid_idents, ["MiB", "GiB", "TiB", "PiB", "EiB", "GitHub", "NaN", "GPLv2", "GPLv3"] => Vec<String>),
    /// Lint: TOO_MANY_ARGUMENTS. The maximum number of argument a function or method can have
    ("too-many-arguments-threshold", too_many_arguments_threshold, 7 => u64),
    /// Lint: TYPE_COMPLEXITY. The maximum complexity a type can have
    ("type-complexity-threshold", type_complexity_threshold, 250 => u64),
    /// Lint: MANY_SINGLE_CHAR_NAMES. The maximum number of single char bindings a scope may have
    ("single-char-binding-names-threshold", max_single_char_names, 5 => u64),
}

/// Read the `toml` configuration file. The function will ignore “File not found” errors iif
/// `!must_exist`, in which case, it will return the default configuration.
/// In case of error, the function tries to continue as much as possible.
pub fn read(path: &str, must_exist: bool) -> (Conf, Vec<Error>) {
    let mut conf = Conf::default();
    let mut errors = Vec::new();

    let file = match fs::File::open(path) {
        Ok(mut file) => {
            let mut buf = String::new();

            if let Err(err) = file.read_to_string(&mut buf) {
                errors.push(err.into());
                return (conf, errors);
            }

            buf
        }
        Err(ref err) if !must_exist && err.kind() == io::ErrorKind::NotFound => {
            return (conf, errors);
        }
        Err(err) => {
            errors.push(err.into());
            return (conf, errors);
        }
    };

    let mut parser = toml::Parser::new(&file);
    let toml = if let Some(toml) = parser.parse() {
        toml
    } else {
        errors.push(Error::TomlError(parser.errors));
        return (conf, errors);
    };

    for (key, value) in toml {
        if let Err(err) = conf.set(key, value) {
            errors.push(err);
        }
    }

    (conf, errors)
}
