//! Test that stepping through a simple program with a debugger one line at a
//! time works intuitively, e.g. that `next` takes you to the next source line.
//! Regression test for <https://github.com/rust-lang/rust/issues/33013>.

//@ ignore-aarch64: Doesn't work yet.
//@ ignore-loongarch64: Doesn't work yet.
//@ ignore-riscv64: Doesn't work yet.
//@ compile-flags: -g

// gdb-command: run
// FIXME(#97083): Should we be able to break on initialization of zero-sized types?
// FIXME(#97083): Right now the first breakable line is:
// gdb-check:   let mut c = 27;
// gdb-command: next
// gdb-check:   let d = c = 99;
// gdb-command: next
// FIXME(#33013): gdb-check:   let e = "hi bob";
// FIXME(#33013): gdb-command: next
// FIXME(#33013): gdb-check:   let f = b"hi bob";
// FIXME(#33013): gdb-command: next
// FIXME(#33013): gdb-check:   let g = b'9';
// FIXME(#33013): gdb-command: next
// FIXME(#33013): gdb-check:   let h = ["whatever"; 8];
// FIXME(#33013): gdb-command: next
// gdb-check:   let i = [1,2,3,4];
// gdb-command: next
// gdb-check:   let j = (23, "hi");
// gdb-command: next
// gdb-check:   let k = 2..3;
// gdb-command: next
// gdb-check:   let l = &i[k];
// gdb-command: next
// gdb-check:   let m: *const() = &a;

fn main () {
    let a = (); // #break
    let b : [i32; 0] = [];
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
