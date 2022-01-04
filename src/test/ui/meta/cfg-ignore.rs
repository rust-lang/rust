// This test checks that the "ignore-cfg" compiletest option matches
// up with rustc's builtin evaluation of cfgs.
//
// ignore-cfg: target_family=unix
// check-pass

#[cfg(target_family = "unix")]
compile_error!("this shouldn't be compiled on unix");

fn main() {}
