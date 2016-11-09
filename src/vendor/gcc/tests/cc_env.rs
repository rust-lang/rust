extern crate tempdir;
extern crate gcc;

use std::env;

mod support;
use support::Test;

#[test]
fn main() {
    ccache();
    distcc();
    ccache_spaces();
}

fn ccache() {
    let test = Test::gnu();
    test.shim("ccache");

    env::set_var("CC", "ccache lol-this-is-not-a-compiler foo");
    test.gcc().file("foo.c").compile("libfoo.a");

    test.cmd(0)
        .must_have("lol-this-is-not-a-compiler foo")
        .must_have("foo.c")
        .must_not_have("ccache");
}

fn ccache_spaces() {
    let test = Test::gnu();
    test.shim("ccache");

    env::set_var("CC", "ccache        lol-this-is-not-a-compiler foo");
    test.gcc().file("foo.c").compile("libfoo.a");
    test.cmd(0).must_have("lol-this-is-not-a-compiler foo");
}

fn distcc() {
    let test = Test::gnu();
    test.shim("distcc");

    env::set_var("CC", "distcc lol-this-is-not-a-compiler foo");
    test.gcc().file("foo.c").compile("libfoo.a");

    test.cmd(0)
        .must_have("lol-this-is-not-a-compiler foo")
        .must_have("foo.c")
        .must_not_have("distcc");
}
