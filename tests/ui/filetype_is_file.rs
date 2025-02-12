#![allow(clippy::needless_if)]
#![warn(clippy::filetype_is_file)]

fn main() -> std::io::Result<()> {
    use std::fs;
    use std::ops::BitOr;

    // !filetype.is_dir()
    if fs::metadata("foo.txt")?.file_type().is_file() {
        //~^ filetype_is_file

        // read file
    }

    // positive of filetype.is_dir()
    if !fs::metadata("foo.txt")?.file_type().is_file() {
        //~^ filetype_is_file

        // handle dir
    }

    // false positive of filetype.is_dir()
    if !fs::metadata("foo.txt")?.file_type().is_file().bitor(true) {
        //~^ filetype_is_file

        // ...
    }

    Ok(())
}
