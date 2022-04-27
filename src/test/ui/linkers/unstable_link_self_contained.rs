// check-fail
// revisions: crt auto linker all
// [crt] compile-flags: -C link-self-contained=crt
// [auto] compile-flags: -C link-self-contained=auto
// [linker] compile-flags: -C link-self-contained=linker
// [all] compile-flags: -C link-self-contained=all

// Test ensuring that the unstable values of the stable `-C link-self-contained` flag
// require using `-Z unstable options`

fn main() {}
