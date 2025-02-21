//@no-rustfix

#![allow(clippy::needless_raw_string_hashes)]
#![warn(clippy::join_absolute_paths)]

use std::path::{Path, PathBuf};

fn main() {
    let path = Path::new("/bin");
    path.join("/sh");
    //~^ join_absolute_paths

    let path = Path::new("C:\\Users");
    path.join("\\user");
    //~^ join_absolute_paths

    let path = PathBuf::from("/bin");
    path.join("/sh");
    //~^ join_absolute_paths

    let path = PathBuf::from("/bin");
    path.join(r#"/sh"#);
    //~^ join_absolute_paths

    let path: &[&str] = &["/bin"];
    path.join("/sh");

    let path = Path::new("/bin");
    path.join("sh");
}
