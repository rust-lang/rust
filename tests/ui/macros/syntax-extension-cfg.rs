//@ run-pass
//@ compile-flags: --cfg foo --cfg qux="foo"
//@ compile-flags: --check-cfg=cfg(foo) --check-cfg=cfg(qux,values("foo"))

pub fn main() {
    // check
    if ! cfg!(foo) { panic!() }
    if   cfg!(not(foo)) { panic!() }

    if ! cfg!(qux="foo") { panic!() }
    if   cfg!(not(qux="foo")) { panic!() }

    if ! cfg!(all(foo, qux="foo")) { panic!() }
    if   cfg!(not(all(foo, qux="foo"))) { panic!() }
    if   cfg!(all(not(all(foo, qux="foo")))) { panic!() }

    if cfg!(FALSE) { panic!() }
    if cfg!(all(FALSE, foo, qux="foo")) { panic!() }
    if cfg!(all(FALSE, foo, qux="foo")) { panic!() }
    if ! cfg!(any(FALSE, foo)) { panic!() }

    if ! cfg!(not(FALSE)) { panic!() }
    if ! cfg!(all(not(FALSE), foo, qux="foo")) { panic!() }
}
