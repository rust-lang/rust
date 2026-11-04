//! First time setup of a dev environment
//!
//! These are build-and-run steps for `./x.py setup`, which allows quickly setting up the directory
//! for modifying, building, and running the compiler and library. Running arbitrary configuration
//! allows setting up things that cannot be simply captured inside the bootstrap.toml, in addition to
//! leading people away from manually editing most of the bootstrap.toml values.

use std::io;
use std::io::Write;
use std::ops::ControlFlow::{self, *};

mod profile;

mod bootstrap_toml;
mod editor;
mod hook;
mod link;

pub use bootstrap_toml::BootstrapToml;
pub use editor::Editor;
pub use hook::Hook;
pub use link::Link;
pub use profile::Profile;

/// Prompt a user for a answer, looping until they enter an accepted input
/// (one for which `f` returns [`Break`]).
fn prompt_user<R>(prompt: &str, f: impl Fn(&str) -> ControlFlow<R>) -> io::Result<R> {
    let mut buf = String::new();
    loop {
        print!("{prompt} ");
        io::stdout().flush()?;

        io::stdin().read_line(&mut buf)?;
        let input = buf.trim().to_lowercase();

        match f(&input) {
            Break(r) => break Ok(r),
            Continue(()) => {
                buf.clear();

                eprintln!("ERROR: unrecognized option '{}'", input.trim());
                eprintln!("NOTE: press Ctrl+C to exit");
            }
        };
    }
}
