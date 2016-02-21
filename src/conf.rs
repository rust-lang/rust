use std::{fmt, fs, io};
use std::io::Read;
use syntax::{ast, codemap, ptr};
use syntax::parse::token;
use toml;

/// Get the configuration file from arguments.
pub fn conf_file(args: &[ptr::P<ast::MetaItem>]) -> Result<Option<token::InternedString>, (&'static str, codemap::Span)> {
    for arg in args {
        match arg.node {
            ast::MetaItemKind::Word(ref name) | ast::MetaItemKind::List(ref name, _) => {
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
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Error from reading a configuration file.
#[derive(Debug)]
pub enum ConfError {
    IoError(io::Error),
    TomlError(Vec<toml::ParserError>),
    TypeError(&'static str, &'static str, &'static str),
    UnknownKey(String),
}

impl fmt::Display for ConfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ConfError::IoError(ref err) => {
                err.fmt(f)
            }
            ConfError::TomlError(ref errs) => {
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
            ConfError::TypeError(ref key, ref expected, ref got) => {
                write!(f, "`{}` is expected to be a `{}` but is a `{}`", key, expected, got)
            }
            ConfError::UnknownKey(ref key) => {
                write!(f, "unknown key `{}`", key)
            }
        }
    }
}

impl From<io::Error> for ConfError {
    fn from(e: io::Error) -> Self {
        ConfError::IoError(e)
    }
}

macro_rules! define_Conf {
    ($(($toml_name: tt, $rust_name: ident, $default: expr, $ty: ident),)+) => {
        /// Type used to store lint configuration.
        pub struct Conf {
            $(pub $rust_name: $ty,)+
        }

        impl Default for Conf {
            fn default() -> Conf {
                Conf {
                    $($rust_name: $default,)+
                }
            }
        }

        impl Conf {
            /// Set the property `name` (which must be the `toml` name) to the given value
            #[allow(cast_sign_loss)]
            fn set(&mut self, name: String, value: toml::Value) -> Result<(), ConfError> {
                match name.as_str() {
                    $(
                        define_Conf!(PAT $toml_name) => {
                            if let Some(value) = define_Conf!(CONV $ty, value) {
                                self.$rust_name = value;
                            }
                            else {
                                return Err(ConfError::TypeError(define_Conf!(EXPR $toml_name),
                                                                stringify!($ty),
                                                                value.type_str()));
                            }
                        },
                    )+
                    _ => {
                        return Err(ConfError::UnknownKey(name));
                    }
                }

                Ok(())
            }
        }
    };

    // hack to convert tts
    (PAT $pat: pat) => { $pat };
    (EXPR $e: expr) => { $e };

    // how to read the value?
    (CONV i64, $value: expr) => { $value.as_integer() };
    (CONV u64, $value: expr) => { $value.as_integer().iter().filter_map(|&i| if i >= 0 { Some(i as u64) } else { None }).next() };
    (CONV String, $value: expr) => { $value.as_str().map(Into::into) };
    (CONV StringVec, $value: expr) => {{
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
}

/// To keep the `define_Conf!` macro simple
pub type StringVec = Vec<String>;

define_Conf! {
    ("blacklisted-names", blacklisted_names, vec!["foo".to_owned(), "bar".to_owned(), "baz".to_owned()], StringVec),
    ("cyclomatic-complexity-threshold", cyclomatic_complexity_threshold, 25, u64),
    ("too-many-arguments-threshold", too_many_arguments_threshold, 6, u64),
    ("type-complexity-threshold", type_complexity_threshold, 250, u64),
}

/// Read the `toml` configuration file. The function will ignore “File not found” errors iif
/// `!must_exist`, in which case, it will return the default configuration.
pub fn read_conf(path: &str, must_exist: bool) -> Result<Conf, ConfError> {
    let mut conf = Conf::default();

    let file = match fs::File::open(path) {
        Ok(mut file) => {
            let mut buf = String::new();
            try!(file.read_to_string(&mut buf));
            buf
        }
        Err(ref err) if !must_exist && err.kind() == io::ErrorKind::NotFound => {
            return Ok(conf);
        }
        Err(err) => {
            return Err(err.into());
        }
    };

    let mut parser = toml::Parser::new(&file);
    let toml = if let Some(toml) = parser.parse() {
        toml
    }
    else {
        return Err(ConfError::TomlError(parser.errors));
    };

    for (key, value) in toml {
        try!(conf.set(key, value));
    }

    Ok(conf)
}
