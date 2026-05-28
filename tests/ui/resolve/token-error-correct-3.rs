// Test that we do some basic error correction in the tokeniser (and don't spew
// too many bogus errors).

pub mod raw {
    use std::{io, fs};
    use std::path::Path;

    pub fn ensure_dir_exists<P: AsRef<Path>, F: FnOnce(&Path)>(path: P,
                                                               callback: F)
                                                               -> io::Result<bool> {
        if !is_directory(path.as_ref()) {
            callback(path.as_ref();
            fs::create_dir_all(path.as_ref()).map(|()| true)
        } else {  //~ ERROR mismatched closing delimiter
            Ok(false);
        }

        panic!();
    }
}

fn main() {}
