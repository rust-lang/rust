//@ revisions: a b c d e f g
//@ ignore-parallel-frontend option conflicts
//@ compile-flags: -Z unstable-options

//@[a] compile-flags: -j 1 --jobs-frontend 2
//@[b] compile-flags: --jobs 1 --jobs-frontend 2
//[a,b]~? ERROR `--jobs-frontend` cannot be larger than `--jobs`

//@[c] compile-flags: --jobs 1 --jobs-backend 2
//[c]~? ERROR `--jobs-backend` cannot be larger than `--jobs`

//@[d] compile-flags: --jobs 1 --jobs-linker 2
//[d]~? ERROR `--jobs-linker` cannot be larger than `--jobs`

//@[e] compile-flags: --jobs 1 -Zthreads=2
//[e]~? ERROR `-Zthreads` cannot be larger than `--jobs`

//@[f] compile-flags: --jobs-frontend 2 -Zthreads=2
//[f]~? ERROR cannot use both `--jobs-frontend` and `-Zthreads`

//@[g] compile-flags: --jobs 1 --jobs 2
//[g]~? RAW Option 'jobs' given more than once

fn main() {}
