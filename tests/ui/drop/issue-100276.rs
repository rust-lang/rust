//@ check-pass
//@ compile-flags: -Z validate-mir
//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] compile-flags: -Z lint-mir
//@ [edition2024] edition: 2024

#![cfg_attr(edition2021, feature(let_chains))]

fn let_chains(entry: std::io::Result<std::fs::DirEntry>) {
    if let Ok(entry) = entry
        && let Some(s) = entry.file_name().to_str()
        && s.contains("")
    {}
}

fn main() {}
