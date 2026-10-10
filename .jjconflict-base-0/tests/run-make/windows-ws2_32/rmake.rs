//@ only-msvc

// Tests that WS2_32.dll is not unnecessarily linked, see issue #85441

use run_make_support::object::read::Object;
use run_make_support::object::{self};
use run_make_support::{rfs, rustc};

fn main() {
    rustc().input("empty.rs").run();
    rustc().input("tcp.rs").run();

    assert!(!links_ws2_32("empty.exe"));
    assert!(links_ws2_32("tcp.exe"));
}

fn links_ws2_32(exe: &str) -> bool {
    let binary_data = rfs::read(exe);
    let file = object::File::parse(&*binary_data).unwrap();
    for import in file.imports().unwrap() {
        if import.library().eq_ignore_ascii_case(b"WS2_32.dll") {
            return true;
        }
    }
    false
}
