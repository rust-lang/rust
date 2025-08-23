// Check that we do not have `profile.*` sections in our `Cargo.toml` files,
// as this causes warnings when run from the compiler repository which includes
// Clippy in a workspace.
//
// Those sections can be put into `.cargo/config.toml` which will be read
// when commands are issued from the top-level Clippy directory, outside of
// a workspace.

use std::fs::File;
use std::io::{self, BufRead as _};
use walkdir::WalkDir;

#[test]
fn no_profile_in_cargo_toml() {
    // This check could parse `Cargo.toml` using a TOML deserializer, but in practice
    // profile sections would be added at the beginning of a line as `[profile.*]`, so
    // keep it fast and simple.
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_str() == Some("Cargo.toml"))
    {
        for line in io::BufReader::new(File::open(entry.path()).unwrap())
            .lines()
            .map(Result::unwrap)
        {
            if line.starts_with("[profile.") {
                eprintln!("Profile section `{line}` found in file `{}`.", entry.path().display());
                eprintln!("Use `.cargo/config.toml` for profiles specific to the standalone Clippy repository.");
                panic!("Profile section found in `Cargo.toml`");
            }
        }
    }
}
