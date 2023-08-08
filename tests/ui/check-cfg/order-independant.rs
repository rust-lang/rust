// check-pass
// revisions: names_before names_after
// compile-flags: -Z unstable-options
// compile-flags: --check-cfg=names(names_before,names_after)
// [names_before]compile-flags: --check-cfg=names(a) --check-cfg=values(a,"b")
// [names_after]compile-flags: --check-cfg=values(a,"b") --check-cfg=names(a)

#[cfg(a)]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

#[cfg(a = "unk")]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

fn main() {}
