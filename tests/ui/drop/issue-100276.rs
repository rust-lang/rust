//@ check-pass
//@ compile-flags: -Z lint-mir -Z validate-mir
//@ edition: 2024

fn let_chains(entry: std::io::Result<std::fs::DirEntry>) {
    if let Ok(entry) = entry
        && let Some(s) = entry.file_name().to_str()
        && s.contains("")
    {}
}

fn main() {}
