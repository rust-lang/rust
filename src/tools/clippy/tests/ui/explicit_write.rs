// run-rustfix
#![allow(unused_imports)]
#![warn(clippy::explicit_write)]

fn stdout() -> String {
    String::new()
}

fn stderr() -> String {
    String::new()
}

macro_rules! one {
    () => {
        1
    };
}

fn main() {
    // these should warn
    {
        use std::io::Write;
        write!(std::io::stdout(), "test").unwrap();
        write!(std::io::stderr(), "test").unwrap();
        writeln!(std::io::stdout(), "test").unwrap();
        writeln!(std::io::stderr(), "test").unwrap();
        std::io::stdout().write_fmt(format_args!("test")).unwrap();
        std::io::stderr().write_fmt(format_args!("test")).unwrap();

        // including newlines
        writeln!(std::io::stdout(), "test\ntest").unwrap();
        writeln!(std::io::stderr(), "test\ntest").unwrap();

        let value = 1;
        writeln!(std::io::stderr(), "with {}", value).unwrap();
        writeln!(std::io::stderr(), "with {} {}", 2, value).unwrap();
        writeln!(std::io::stderr(), "with {value}").unwrap();
        writeln!(std::io::stderr(), "macro arg {}", one!()).unwrap();
    }
    // these should not warn, different destination
    {
        use std::fmt::Write;
        let mut s = String::new();
        write!(s, "test").unwrap();
        write!(s, "test").unwrap();
        writeln!(s, "test").unwrap();
        writeln!(s, "test").unwrap();
        s.write_fmt(format_args!("test")).unwrap();
        s.write_fmt(format_args!("test")).unwrap();
        write!(stdout(), "test").unwrap();
        write!(stderr(), "test").unwrap();
        writeln!(stdout(), "test").unwrap();
        writeln!(stderr(), "test").unwrap();
        stdout().write_fmt(format_args!("test")).unwrap();
        stderr().write_fmt(format_args!("test")).unwrap();
    }
    // these should not warn, no unwrap
    {
        use std::io::Write;
        std::io::stdout().write_fmt(format_args!("test")).expect("no stdout");
        std::io::stderr().write_fmt(format_args!("test")).expect("no stderr");
    }
}
