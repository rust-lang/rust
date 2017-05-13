//! Simple script to verify the coding style of this library
//!
//! ## How to run
//!
//! The first argument to this script is the directory to run on, so running
//! this script should be as simple as:
//!
//! ```notrust
//! rustc ci/style.rs
//! ./style src
//! ```
//!
//! ## Guidelines
//!
//! The current style is:
//!
//! * No trailing whitespace
//! * No tabs
//! * 80-character lines
//! * `extern` instead of `extern "C"`
//! * Specific module layout:
//!     1. use directives
//!     2. typedefs
//!     3. structs
//!     4. constants
//!     5. f! { ... } functions
//!     6. extern functions
//!     7. modules + pub use
//!
//! Things not verified:
//!
//! * alignment
//! * 4-space tabs
//! * leading colons on paths

use std::env;
use std::fs;
use std::io::prelude::*;
use std::path::Path;

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

fn main() {
    let arg = env::args().skip(1).next().unwrap_or(".".to_string());

    let mut errors = Errors { errs: false };
    walk(Path::new(&arg), &mut errors);

    if errors.errs {
        panic!("found some lint errors");
    } else {
        println!("good style!");
    }
}

fn walk(path: &Path, err: &mut Errors) {
    for entry in t!(path.read_dir()).map(|e| t!(e)) {
        let path = entry.path();
        if t!(entry.file_type()).is_dir() {
            walk(&path, err);
            continue
        }

        let name = entry.file_name().into_string().unwrap();
        match &name[..] {
            n if !n.ends_with(".rs") => continue,

            "dox.rs" |
            "lib.rs" |
            "macros.rs" => continue,

            _ => {}
        }

        let mut contents = String::new();
        t!(t!(fs::File::open(&path)).read_to_string(&mut contents));

        check_style(&contents, &path, err);
    }
}

struct Errors {
    errs: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Start,
    Imports,
    Typedefs,
    Structs,
    Constants,
    FunctionDefinitions,
    Functions,
    Modules,
}

fn check_style(file: &str, path: &Path, err: &mut Errors) {
    let mut state = State::Start;
    let mut s_macros = 0;
    let mut f_macros = 0;
    let mut prev_blank = false;

    for (i, line) in file.lines().enumerate() {
        if line == "" {
            if prev_blank {
                err.error(path, i, "double blank line");
            }
            prev_blank = true;
        } else {
            prev_blank = false;
        }
        if line != line.trim_right() {
            err.error(path, i, "trailing whitespace");
        }
        if line.contains("\t") {
            err.error(path, i, "tab character");
        }
        if line.len() > 80 {
            err.error(path, i, "line longer than 80 chars");
        }
        if line.contains("extern \"C\"") {
            err.error(path, i, "use `extern` instead of `extern \"C\"");
        }
        if line.contains("#[cfg(") && !line.contains(" if ") {
            if state != State::Structs {
                err.error(path, i, "use cfg_if! and submodules \
                                    instead of #[cfg]");
            }
        }

        let line = line.trim_left();
        let is_pub = line.starts_with("pub ");
        let line = if is_pub {&line[4..]} else {line};

        let line_state = if line.starts_with("use ") {
            if is_pub {
                State::Modules
            } else {
                State::Imports
            }
        } else if line.starts_with("const ") {
            State::Constants
        } else if line.starts_with("type ") {
            State::Typedefs
        } else if line.starts_with("s! {") {
            s_macros += 1;
            State::Structs
        } else if line.starts_with("f! {") {
            f_macros += 1;
            State::FunctionDefinitions
        } else if line.starts_with("extern ") {
            State::Functions
        } else if line.starts_with("mod ") {
            State::Modules
        } else {
            continue
        };

        if state as usize > line_state as usize {
            err.error(path, i, &format!("{} found after {} when \
                                         it belongs before",
                                        line_state.desc(), state.desc()));
        }

        if f_macros == 2 {
            f_macros += 1;
            err.error(path, i, "multiple f! macros in one module");
        }
        if s_macros == 2 {
            s_macros += 1;
            err.error(path, i, "multiple s! macros in one module");
        }

        state = line_state;
    }
}

impl State {
    fn desc(&self) -> &str {
        match *self {
            State::Start => "start",
            State::Imports => "import",
            State::Typedefs => "typedef",
            State::Structs => "struct",
            State::Constants => "constant",
            State::FunctionDefinitions => "function definition",
            State::Functions => "extern function",
            State::Modules => "module",
        }
    }
}

impl Errors {
    fn error(&mut self, path: &Path, line: usize, msg: &str) {
        self.errs = true;
        println!("{}:{} - {}", path.display(), line + 1, msg);
    }
}
