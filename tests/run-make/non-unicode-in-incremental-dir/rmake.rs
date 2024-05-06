use run_make_support::{rustc, tmp_dir};

fn main() {
    #[cfg(unix)]
    let non_unicode: &std::ffi::OsStr = std::os::unix::ffi::OsStrExt::from_bytes(&[0xFF]);
    #[cfg(windows)]
    let non_unicode: std::ffi::OsString = std::os::windows::ffi::OsStringExt::from_wide(&[0xD800]);
    match std::fs::create_dir(tmp_dir().join(&non_unicode)) {
        // If an error occurs, check if creating a directory with a valid Unicode name would
        // succeed.
        Err(e) if std::fs::create_dir(tmp_dir().join("valid_unicode")).is_ok() => {
            // Filesystem doesn't appear support non-Unicode paths.
            return;
        }
        Err(e) => panic!("error creating non-Unicode directory: {e}"),
        _ => {}
    }
    let incr_dir = tmp_dir().join("incr-dir");
    rustc().input("foo.rs").incremental(&incr_dir).run();
    for crate_dir in std::fs::read_dir(&incr_dir).unwrap() {
        std::fs::create_dir(crate_dir.unwrap().path().join(&non_unicode)).unwrap();
    }
    rustc().input("foo.rs").incremental(&incr_dir).run();
}
