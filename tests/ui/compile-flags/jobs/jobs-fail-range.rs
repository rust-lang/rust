//@ revisions: a b c d e f g h i j k l
//@ compile-flags: -Z unstable-options

//@[a] compile-flags: -j 256
//@[b] compile-flags: -j -1
//@[c] compile-flags: -j nonsense
//@[d] compile-flags: --jobs nonsense
//[a,b,c,d]~? ERROR `--jobs`: expected a number from 0 to 255 or `sync`

//@[e] compile-flags: --jobs-frontend nonsense
//[e]~? ERROR `--jobs-frontend`: expected a number from 0 to 255 or `sync`

//@[f] compile-flags: --jobs-backend nonsense
//[f]~? ERROR `--jobs-backend`: expected a number from 0 to 255 or `sync`

//@[g] compile-flags: --jobs-linker nonsense
//[g]~? ERROR `--jobs-linker`: expected a number from 0 to 255 or `sync`

//@[h] compile-flags: -Zthreads=nonsense
//[h]~? ERROR `-Zthreads`: expected a number from 0 to 255 or `sync`

//@[i] compile-flags: -j
//@[j] compile-flags: --jobs
//@[k] compile-flags: -Zthreads
//@[l] compile-flags: -Zno-parallel-backend
//[i]~? RAW Argument to option 'j' missing
//[j]~? RAW Argument to option 'jobs' missing
//[k]~? ERROR unstable option `threads` requires a string
//[l]~? ERROR `-Zno-parallel-backend` is removed, use `--jobs-backend=1` instead

fn main() {}
