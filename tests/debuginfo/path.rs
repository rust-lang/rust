//@ ignore-gdb

//@ compile-flags:-g

// === LLDB TESTS =================================================================================

// lldb-command:run

// lldb-command:print pathbuf
// lldb-check:[...] "/some/path" { inner = "/some/path" { inner = { inner = size=10 { [0] = '/' [1] = 's' [2] = 'o' [3] = 'm' [4] = 'e' [5] = '/' [6] = 'p' [7] = 'a' [8] = 't' [9] = 'h' } } } }
// lldb-command:po pathbuf
// lldb-check:"/some/path"
// lldb-command:print path
// lldb-check:[...] "/some/path" { data_ptr = [...] length = 10 }
// lldb-command:po path
// lldb-check:"/some/path"

use std::path::Path;

fn main() {
    let path = Path::new("/some/path");
    let pathbuf = path.to_path_buf();

    zzz(); // #break
}

fn zzz() {
    ()
}
