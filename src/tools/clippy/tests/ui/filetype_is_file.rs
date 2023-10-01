#![allow(clippy::needless_if)]
#![warn(clippy::filetype_is_file)]

fn main() -> std::io::Result<()> {
    use std::fs;
    use std::ops::BitOr;

    // !filetype.is_dir()
    if fs::metadata("foo.txt")?.file_type().is_file() {
        //~^ ERROR: `FileType::is_file()` only covers regular files
        // read file
    }

    // positive of filetype.is_dir()
    if !fs::metadata("foo.txt")?.file_type().is_file() {
        //~^ ERROR: `!FileType::is_file()` only denies regular files
        // handle dir
    }

    // false positive of filetype.is_dir()
    if !fs::metadata("foo.txt")?.file_type().is_file().bitor(true) {
        //~^ ERROR: `FileType::is_file()` only covers regular files
        // ...
    }

    Ok(())
}
