// xfail-fast
// aux-build:pub_use_mods_xcrate.rs

extern mod pub_use_mods_xcrate;
use pub_use_mods_xcrate::a::c;

fn main(){}

