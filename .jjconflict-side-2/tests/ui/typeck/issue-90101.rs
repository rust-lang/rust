use std::path::{Path, PathBuf};

fn func(path: impl Into<PathBuf>, code: impl Into<String>) {}

fn main() {
    func(Path::new("hello").to_path_buf().to_string_lossy(), "world")
    //~^ ERROR [E0277]
}
