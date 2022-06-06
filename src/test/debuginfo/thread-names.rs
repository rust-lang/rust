// compile-flags:-g
// We can't set the main thread name on Linux because it renames the process (#97191)
// ignore-linux
// ignore-android
// ignore-dragonfly
// ignore-emscripten
// ignore-freebsd
// ignore-haiku
// ignore-ios
// ignore-netbsd
// ignore-openbsd
// ignore-solaris
// ignore-sgx
// ignore-windows-gnu

// === GDB TESTS ==================================================================================
//
// gdb-command:run
//
// gdb-command:info threads
// gdb-check:  1    Thread [...] [...] "main" [...]
// gdb-check:* 2    Thread [...] [...] "my new thread" [...]

// === LLDB TESTS =================================================================================
//
// lldb-command:run
//
// lldb-command:thread info 1
// lldb-check:thread #1:[...]name = 'main'[...]
// lldb-command:thread info 2
// lldb-check:thread #2:[...]name = 'my new thread'[...]

// === CDB TESTS ==================================================================================
//
// cdb-command:g
//
// cdb-command:~
// cdb-check:   0  Id: [...] Suspend: 1 Teb: [...] Unfrozen "main"
// cdb-check:.  [...]  Id: [...] Suspend: 1 Teb: [...] Unfrozen "my new thread"

use std::thread;

fn main() {
    let handle = thread::Builder::new().name("my new thread".into()).spawn(|| {
        zzz(); // #break
    }).unwrap();

    handle.join().unwrap();
}

fn zzz() {}
