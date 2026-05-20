//@ ignore-gdb

//@ compile-flags:-g

// === LLDB TESTS =================================================================================

//@ lldb-command:run

//@ lldb-command:print pathbuf
//@ lldb-check:[...] "/some/path" [...]
//@ lldb-command:print path
//@ lldb-check:[...] "/some/path" [...]

use std::path::Path;

fn main() {
    let path = Path::new("/some/path");
    let pathbuf = path.to_path_buf();

    zzz(); // #break
}

fn zzz() {
    ()
}
