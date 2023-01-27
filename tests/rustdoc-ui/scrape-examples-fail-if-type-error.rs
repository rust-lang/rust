// check-fail
// compile-flags: -Z unstable-options --scrape-examples-output-path {{build-base}}/t.calls --scrape-examples-target-crate foobar

pub fn foo() {
    INVALID_FUNC();
    //~^ ERROR cannot find function, tuple struct or tuple variant `INVALID_FUNC` in this scope
}
