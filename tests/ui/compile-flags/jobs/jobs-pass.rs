//@ check-pass
//@ revisions: a b c d e f g
//@ ignore-parallel-frontend option conflicts
//@ compile-flags: -Z unstable-options

//@[a] compile-flags: -j 0
//@[b] compile-flags: -j 1
//@[c] compile-flags: -j 255
//@[d] compile-flags: --jobs 16 --jobs-frontend 8 --jobs-backend 4 --jobs-linker 2
//@[e] compile-flags: --jobs 16 -Zthreads=8 --jobs-backend 4
//@[g] compile-flags: -Zthreads=1 -Zthreads=2

fn main() {}
