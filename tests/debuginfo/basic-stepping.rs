//! Test that stepping through a simple program with a debugger one line at a
//! time works intuitively, e.g. that `next` takes you to the next source line.
//! Regression test for <https://github.com/rust-lang/rust/issues/33013>.

//@ ignore-loongarch64: Doesn't work yet.
//@ ignore-riscv64: Doesn't work yet.
//@ ignore-backends: gcc

// Debugger tests need debuginfo
//@ compile-flags: -g

// Some optimization passes _improve_ compile times [1]. So we want to run some
// passes even with `-Copt-level=0`. That means that some of the lines below can
// be optimized away. To make regression testing more robust, we also want to
// run this test with such passes disabled. The solution is to use two
// revisions. One with default `-Copt-level=0` passes, and one "even less
// optimized", with enough optimization passes disabled to keep the maximum
// number of lines steppable.
//
// If `-Zmir-enable-passes=-...` ends up being annoying to maintain, we can try
// switching to `-Zmir-opt-level=0` instead.
//
// [1]: https://github.com/rust-lang/compiler-team/issues/319
//@ revisions: opt-level-0 maximally-steppable
//@ [maximally-steppable] compile-flags: -Zmir-enable-passes=-SingleUseConsts

// === GDB TESTS ===================================================================================

//@ gdb-command: run
//@ gdb-check:   let mut c = 27;
//@ gdb-command: next
//@ gdb-check:   let d = c = 99;
//@ gdb-command: next
//@ [maximally-steppable] gdb-check:   let e = "hi bob";
//@ [maximally-steppable] gdb-command: next
//@ [maximally-steppable] gdb-check:   let f = b"hi bob";
//@ [maximally-steppable] gdb-command: next
//@ [maximally-steppable] gdb-check:   let g = b'9';
//@ [maximally-steppable] gdb-command: next
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
//@ [maximally-steppable] lldb-check:   [...]let e = "hi bob";[...]
//@ [maximally-steppable] lldb-command: next
//@ [maximally-steppable] lldb-command: frame select
//@ [maximally-steppable] lldb-check:   [...]let f = b"hi bob";[...]
//@ [maximally-steppable] lldb-command: next
//@ [maximally-steppable] lldb-command: frame select
//@ [maximally-steppable] lldb-check:   [...]let g = b'9';[...]
//@ [maximally-steppable] lldb-command: next
//@ [maximally-steppable] lldb-command: frame select
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

// === CDB TESTS ===================================================================================

// Enable source line support. See
// https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/-lines--toggle-source-line-support-.
//@ cdb-command: .lines -e
// Display source lines and source line numbers at the command prompt. See
// https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/l---l---set-source-options-.
//@ cdb-command: l+s
// Enter "source mode" so `p` steps source lines and not assembly instructions.
//@ cdb-command: l+t

// `g` means "go". See
// https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/g--go-.
//@ cdb-command: g
// `p` means "step". See
// https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/p--step-.
//@ cdb-command: p
//@ cdb-check:   [...]:     let mut c = 27;
//@ cdb-command: p
//@ cdb-check:   [...]:     let d = c = 99;
//@ [maximally-steppable] cdb-command: p
//@ [maximally-steppable] cdb-check:   [...]:     let e = "hi bob";
//@ [maximally-steppable] cdb-command: p
//@ [maximally-steppable] cdb-check:   [...]:     let f = b"hi bob";
//@ [maximally-steppable] cdb-command: p
//@ [maximally-steppable] cdb-check:   [...]:     let g = b'9';
//@ cdb-command: p
//@ cdb-check:   [...]:     let h = ["whatever"; 8];
//@ cdb-command: p
//@ cdb-check:   [...]:     let i = [1,2,3,4];
//@ cdb-command: p
//@ cdb-check:   [...]:     let j = (23, "hi");
//@ cdb-command: p
//@ cdb-check:   [...]:     let k = 2..3;
//@ cdb-command: p
//@ cdb-check:   [...]:     let l = &i[k];
//@ cdb-command: p
//@ cdb-check:   [...]:     let m: *const() = &a;

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
