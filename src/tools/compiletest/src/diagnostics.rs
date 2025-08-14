//! Collection of diagnostics helpers for `compiletest` *itself*.

#[macro_export]
macro_rules! fatal {
    ($($arg:tt)*) => {
        let status = ::colored::Colorize::bright_red("FATAL: ");
        let status = ::colored::Colorize::bold(status);
        eprint!("{status}");
        eprintln!($($arg)*);
        // This intentionally uses a seemingly-redundant panic to include backtrace location.
        //
        // FIXME: in the long term, we should handle "logic bug in compiletest itself" vs "fatal
        // user error" separately.
        panic!("fatal error");
    };
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        let status = ::colored::Colorize::red("ERROR: ");
        let status = ::colored::Colorize::bold(status);
        eprint!("{status}");
        eprintln!($($arg)*);
    };
}

#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {
        let status = ::colored::Colorize::yellow("WARNING: ");
        let status = ::colored::Colorize::bold(status);
        eprint!("{status}");
        eprintln!($($arg)*);
    };
}

#[macro_export]
macro_rules! help {
    ($($arg:tt)*) => {
        let status = ::colored::Colorize::cyan("HELP: ");
        let status = ::colored::Colorize::bold(status);
        eprint!("{status}");
        eprintln!($($arg)*);
    };
}
