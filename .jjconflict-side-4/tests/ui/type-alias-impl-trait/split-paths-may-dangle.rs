// Regression test for issue #146045 - ensure that the TAIT `SplitPaths` does not
// require the borrowed string to be live.
//@ check-pass
//@ edition:2015

pub fn repro() -> Option<std::path::PathBuf> {
    let unparsed = std::ffi::OsString::new();
    std::env::split_paths(&unparsed).next()
}

fn main() {}
