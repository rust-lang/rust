use rustc::session::Session;

use std::io;
use std::path::{Path, PathBuf};

pub fn find_library(name: &str, search_paths: &[PathBuf], sess: &Session)
                    -> PathBuf {
    // On Windows, static libraries sometimes show up as libfoo.a and other
    // times show up as foo.lib
    let oslibname = format!("{}{}{}",
                            sess.target.target.options.staticlib_prefix,
                            name,
                            sess.target.target.options.staticlib_suffix);
    let unixlibname = format!("lib{}.a", name);

    for path in search_paths {
        debug!("looking for {} inside {:?}", name, path);
        let test = path.join(&oslibname);
        if test.exists() { return test }
        if oslibname != unixlibname {
            let test = path.join(&unixlibname);
            if test.exists() { return test }
        }
    }
    sess.fatal(&format!("could not find native static library `{}`, \
                         perhaps an -L flag is missing?", name));
}

pub trait ArchiveBuilder<'a> {
    fn new(sess: &'a Session, output: &Path, input: Option<&Path>) -> Self;

    fn add_file(&mut self, path: &Path);
    fn remove_file(&mut self, name: &str);
    fn src_files(&mut self) -> Vec<String>;

    fn add_rlib(
        &mut self,
        path: &Path,
        name: &str,
        lto: bool,
        skip_objects: bool,
    ) -> io::Result<()>;
    fn add_native_library(&mut self, name: &str);
    fn update_symbols(&mut self);

    fn build(self);
}
