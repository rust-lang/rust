use rustc::session::Session;

use std::path::PathBuf;

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
