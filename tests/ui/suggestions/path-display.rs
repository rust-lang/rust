use std::path::{Path, PathBuf};

fn main() {
    let path = Path::new("/tmp/foo/bar.txt");
    println!("{}", path);
    //~^ ERROR E0277

    let path = PathBuf::from("/tmp/foo/bar.txt");
    println!("{}", path);
    //~^ ERROR E0277
}
