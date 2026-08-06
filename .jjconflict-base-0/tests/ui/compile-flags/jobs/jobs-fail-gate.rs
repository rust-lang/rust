//@ revisions: a b c d e f

//@[a] compile-flags: -j 2
//@[b] compile-flags: --jobs 2
//@[c] compile-flags: --jobs sync
//[a,b,c]~? RAW the `-Z unstable-options` flag must also be passed to enable the flag `jobs`

//@[d] compile-flags: --jobs-frontend 2
//[d]~? RAW the `-Z unstable-options` flag must also be passed to enable the flag `jobs-frontend`

//@[e] compile-flags: --jobs-backend 2
//[e]~? RAW the `-Z unstable-options` flag must also be passed to enable the flag `jobs-backend`

//@[f] compile-flags: --jobs-linker 2
//[f]~? RAW the `-Z unstable-options` flag must also be passed to enable the flag `jobs-linker`

fn main() {}
