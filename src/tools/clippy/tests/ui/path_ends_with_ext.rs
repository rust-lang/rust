#![warn(clippy::path_ends_with_ext)]
use std::path::Path;

macro_rules! arg {
    () => {
        ".md"
    };
}

fn test(path: &Path) {
    path.ends_with(".md");
    //~^ ERROR: this looks like a failed attempt at checking for the file extension

    // some "extensions" are allowed by default
    path.ends_with(".git");

    // most legitimate "dotfiles" are longer than 3 chars, so we allow them as well
    path.ends_with(".bashrc");

    // argument from expn shouldn't trigger
    path.ends_with(arg!());

    path.ends_with("..");
    path.ends_with("./a");
    path.ends_with(".");
    path.ends_with("");
}

// is_some_and was stabilized in 1.70, so suggest map_or(false, ..) if under that
#[clippy::msrv = "1.69"]
fn under_msv(path: &Path) -> bool {
    path.ends_with(".md")
    //~^ ERROR: this looks like a failed attempt at checking for the file extension
}

fn main() {}
