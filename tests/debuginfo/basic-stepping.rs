//! Test that stepping through a simple program with a debugger one line at a
//! time works intuitively, e.g. that `next` takes you to the next source line.
//! Regression test for <https://github.com/rust-lang/rust/issues/33013>.

//@ ignore-aarch64: Doesn't work yet.
//@ ignore-loongarch64: Doesn't work yet.
//@ ignore-riscv64: Doesn't work yet.
//@ ignore-backends: gcc

// Debugger tests need debuginfo
//@ compile-flags: -g

// FIXME(#128945): SingleUseConsts shouldn't need to be disabled.
//@ revisions: default-mir-passes no-SingleUseConsts-mir-pass
//@ [no-SingleUseConsts-mir-pass] compile-flags: -Zmir-enable-passes=-SingleUseConsts

// === GDB TESTS ===================================================================================

//@ gdb-command: run
//@ gdb-check:   let mut c = 27;
//@ gdb-command: next
//@ gdb-check:   let d = c = 99;
//@ gdb-command: next
//@ [no-SingleUseConsts-mir-pass] gdb-check:   let e = "hi bob";
//@ [no-SingleUseConsts-mir-pass] gdb-command: next
//@ [no-SingleUseConsts-mir-pass] gdb-check:   let f = b"hi bob";
//@ [no-SingleUseConsts-mir-pass] gdb-command: next
//@ [no-SingleUseConsts-mir-pass] gdb-check:   let g = b'9';
//@ [no-SingleUseConsts-mir-pass] gdb-command: next
//@ gdb-check:   let h = ["whatever"; 8];
//@ gdb-command: next
//@ gdb-check:   let i = [1,2,3,4];
//@ gdb-command: next
//@ gdb-check:   let j = (23, "hi");
//@ gdb-command: next
//@ gdb-check:   let k = 2..3;
//@ gdb-command: next
//@ gdb-check:   let l = &i[k];
//@ gdb-command: next
//@ gdb-check:   let m: *const() = &a;

// === LLDB TESTS ==================================================================================

// Unlike gdb, lldb will display 7 lines of context by default. It seems
// impossible to get it down to 1. The best we can do is to show the current
// line and one above. That is not ideal, but it will do for now.
//@ lldb-command: settings set stop-line-count-before 1
//@ lldb-command: settings set stop-line-count-after 0

//@ lldb-command: run
// In `breakpoint_callback()` in `./src/etc/lldb_batchmode.py` we do
// `SetSelectedFrame()`, which causes LLDB to show the current line and one line
// before (since we changed `stop-line-count-before`). Note that
// `normalize_whitespace()` in `lldb_batchmode.py` removes the newlines of the
// output. So the current line and the line before actually ends up on the same
// output line. That's fine.
//@ lldb-check:   [...]let mut c = 27;[...]
//@ lldb-command: next
// From now on we must manually `frame select` to see the current line (and one
// line before).
//@ lldb-command: frame select
//@ lldb-check:   [...]let d = c = 99;[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ [no-SingleUseConsts-mir-pass] lldb-check:   [...]let e = "hi bob";[...]
//@ [no-SingleUseConsts-mir-pass] lldb-command: next
//@ [no-SingleUseConsts-mir-pass] lldb-command: frame select
//@ [no-SingleUseConsts-mir-pass] lldb-check:   [...]let f = b"hi bob";[...]
//@ [no-SingleUseConsts-mir-pass] lldb-command: next
//@ [no-SingleUseConsts-mir-pass] lldb-command: frame select
//@ [no-SingleUseConsts-mir-pass] lldb-check:   [...]let g = b'9';[...]
//@ [no-SingleUseConsts-mir-pass] lldb-command: next
//@ [no-SingleUseConsts-mir-pass] lldb-command: frame select
//@ lldb-check:   [...]let h = ["whatever"; 8];[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ lldb-check:   [...]let i = [1,2,3,4];[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ lldb-check:   [...]let j = (23, "hi");[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ lldb-check:   [...]let k = 2..3;[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ lldb-check:   [...]let l = &i[k];[...]
//@ lldb-command: next
//@ lldb-command: frame select
//@ lldb-check:   [...]let m: *const() = &a;[...]

#![allow(unused_assignments, unused_variables)]

fn main () {
    let a = (); // #break
    let b : [i32; 0] = [];
    // FIXME(#97083): Should we be able to break on initialization of zero-sized types?
    // FIXME(#97083): Right now the first breakable line is:
    let mut c = 27;
    let d = c = 99;
    let e = "hi bob";
    let f = b"hi bob";
    let g = b'9';
    let h = ["whatever"; 8];
    let i = [1,2,3,4];
    let j = (23, "hi");
    let k = 2..3;
    let l = &i[k];
    let m: *const() = &a;
}
