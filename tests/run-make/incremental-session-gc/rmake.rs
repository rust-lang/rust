//! Successful sequential builds should retain only the newest incremental session.
//! The current session must participate in garbage collection once it is finalized.

use std::path::PathBuf;

use run_make_support::{rfs, rustc, shallow_find_directories};

fn main() {
    let compile = || {
        rustc().input("empty.rs").crate_type("rlib").emit("metadata").incremental("incr").run();
    };

    compile();
    let mut previous = session_dir();
    rfs::write(previous.join("sentinel"), "previous session");

    for _ in 0..2 {
        compile();
        let current = session_dir();
        assert_ne!(previous, current);
        assert!(!previous.exists(), "superseded session was not collected: {previous:?}");
        assert_eq!(rfs::read_to_string(current.join("sentinel")), "previous session");
        previous = current;
    }
}

fn session_dir() -> PathBuf {
    let crate_dirs = shallow_find_directories("incr", |_| true);
    assert_eq!(crate_dirs.len(), 1);
    let sessions = shallow_find_directories(&crate_dirs[0], |_| true);
    assert_eq!(sessions.len(), 1, "expected only the newest completed session: {sessions:?}");
    sessions.into_iter().next().unwrap()
}
