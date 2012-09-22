// compile-flags: --test --cfg ignorecfg
// xfail-fast
// xfail-pretty

extern mod std;

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
    // Pull the tests out of the secreturn test module
    let tests = __test::tests();

    let shouldignore = option::get(
        &vec::find(tests, |t| t.name == ~"shouldignore" ));
    assert shouldignore.ignore == true;

    let shouldnotignore = option::get(
        &vec::find(tests, |t| t.name == ~"shouldnotignore" ));
    assert shouldnotignore.ignore == false;
}