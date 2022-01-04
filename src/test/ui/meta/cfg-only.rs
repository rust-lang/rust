// This test checks that the "ignore-cfg" compiletest option matches
// up with rustc's builtin evaluation of cfgs.
//
// only-cfg: target_family=unix
// check-pass

#[cfg(not(target_family = "unix"))]
compile_error!("this should only be compiled on unix");

fn main() {}
