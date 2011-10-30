// compile-flags: --test --cfg ignorecfg
// xfail-fast
// xfail-pretty

use std;
import std::option;
import std::vec;

#[test]
#[ignore(cfg(ignorecfg))]
fn shouldignore() {
}

#[test]
#[ignore(cfg(noignorecfg))]
fn shouldnotignore() {
}

#[test]
fn checktests() {
    // Pull the tests out of the secret test module
    let tests = __test::tests();

    let shouldignore = option::get(
        vec::find({|t| t.name == "shouldignore"}, tests));
    assert shouldignore.ignore == true;

    let shouldnotignore = option::get(
        vec::find({|t| t.name == "shouldnotignore"}, tests));
    assert shouldnotignore.ignore == false;
}