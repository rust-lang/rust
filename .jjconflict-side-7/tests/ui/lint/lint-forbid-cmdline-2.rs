//@ compile-flags: -F dead_code

#[allow(unused)]
//~^ ERROR allow(unused) incompatible with previous forbid [E0453]
//~| NOTE overruled by previous forbid
//~| NOTE `forbid` lint level was set on command line (`-F dead_code`)
fn main() {
}
