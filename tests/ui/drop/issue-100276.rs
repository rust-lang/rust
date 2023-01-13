// check-pass
// compile-flags: -Z validate-mir
#![feature(let_chains)]

fn let_chains(entry: std::io::Result<std::fs::DirEntry>) {
    if let Ok(entry) = entry
        && let Some(s) = entry.file_name().to_str()
        && s.contains("")
    {}
}

fn main() {}
