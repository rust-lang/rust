//! First time setup of a dev environment
//!
//! These are build-and-run steps for `./x.py setup`, which allows quickly setting up the directory
//! for modifying, building, and running the compiler and library. Running arbitrary configuration
//! allows setting up things that cannot be simply captured inside the bootstrap.toml, in addition to
//! leading people away from manually editing most of the bootstrap.toml values.

use std::io;
use std::io::Write;

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

#[derive(PartialEq)]
enum PromptResult {
    Yes,   // y/Y/yes
    No,    // n/N/no
    Print, // p/P/print
}

/// Prompt a user for a answer, looping until they enter an accepted input or nothing
fn prompt_user(prompt: &str) -> io::Result<Option<PromptResult>> {
    let mut input = String::new();
    loop {
        print!("{prompt} ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => return Ok(Some(PromptResult::Yes)),
            "n" | "no" => return Ok(Some(PromptResult::No)),
            "p" | "print" => return Ok(Some(PromptResult::Print)),
            "" => return Ok(None),
            _ => {
                eprintln!("ERROR: unrecognized option '{}'", input.trim());
                eprintln!("NOTE: press Ctrl+C to exit");
            }
        };
    }
}
