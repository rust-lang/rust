//! This build script ensures that clippy is not compiled with an
//! incompatible version of rust. It will panic with a descriptive
//! error message instead.
//!
//! We specifially want to ensure that clippy is only built with a
//! rustc version that is newer or equal to the one specified in the
//! `min_version.txt` file.
//!
//! `min_version.txt` is in the repo but also in the `.gitignore` to
//! make sure that it is not updated manually by accident. Only CI
//! should update that file.
//!
//! This build script was originally taken from the Rocket web framework:
//! https://github.com/SergioBenitez/Rocket

use ansi_term::Colour::Red;
use rustc_version::{version_meta, version_meta_for, Channel, Version, VersionMeta};
use std::env;

fn main() {
    check_rustc_version();

    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
    // Don't rebuild even if nothing changed
    println!("cargo:rerun-if-changed=build.rs");
}

fn check_rustc_version() {
    let string = include_str!("min_version.txt");
    let min_version_meta = version_meta_for(string).expect("Could not parse version string in min_version.txt");
    let current_version_meta = version_meta().expect("Could not retrieve current rustc version information from ENV");

    let min_version = min_version_meta.clone().semver;
    let min_date_str = min_version_meta
        .clone()
        .commit_date
        .expect("min_version.txt does not contain a rustc commit date");

    // Dev channel (rustc built from git) does not have any date or commit information in rustc -vV
    // `current_version_meta.commit_date` would crash, so we return early here.
    if current_version_meta.channel == Channel::Dev {
        return;
    }

    let current_version = current_version_meta.clone().semver;
    let current_date_str = current_version_meta
        .clone()
        .commit_date
        .expect("current rustc version information does not contain a rustc commit date");

    let print_version_err = |version: &Version, date: &str| {
        eprintln!(
            "> {} {}. {} {}.\n",
            "Installed rustc version is:",
            format!("{} ({})", version, date),
            "Minimum required rustc version:",
            format!("{} ({})", min_version, min_date_str)
        );
    };

    if !correct_channel(&current_version_meta) {
        eprintln!(
            "\n{} {}",
            Red.bold().paint("error:"),
            "clippy requires a nightly version of Rust."
        );
        print_version_err(&current_version, &*current_date_str);
        eprintln!(
            "{}{}{}",
            "See the README (", "https://github.com/rust-lang-nursery/rust-clippy#usage", ") for more information."
        );
        panic!("Aborting compilation due to incompatible compiler.")
    }

    let current_date = str_to_ymd(&current_date_str).unwrap();
    let min_date = str_to_ymd(&min_date_str).unwrap();

    if current_date < min_date {
        eprintln!(
            "\n{} {}",
            Red.bold().paint("error:"),
            "clippy does not support this version of rustc nightly."
        );
        eprintln!(
            "> {}{}{}",
            "Use `", "rustup update", "` or your preferred method to update Rust."
        );
        print_version_err(&current_version, &*current_date_str);
        panic!("Aborting compilation due to incompatible compiler.")
    }
}

fn correct_channel(version_meta: &VersionMeta) -> bool {
    match version_meta.channel {
        Channel::Stable | Channel::Beta => false,
        Channel::Nightly | Channel::Dev => true,
    }
}

/// Convert a string of %Y-%m-%d to a single u32 maintaining ordering.
fn str_to_ymd(ymd: &str) -> Option<u32> {
    let ymd: Vec<u32> = ymd.split("-").filter_map(|s| s.parse::<u32>().ok()).collect();
    if ymd.len() != 3 {
        return None;
    }

    let (y, m, d) = (ymd[0], ymd[1], ymd[2]);
    Some((y << 9) | (m << 5) | d)
}
