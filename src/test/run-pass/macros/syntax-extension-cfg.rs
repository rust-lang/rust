// run-pass
// compile-flags: --cfg foo --cfg qux="foo"


pub fn main() {
    // check
    if ! cfg!(foo) { panic!() }
    if   cfg!(not(foo)) { panic!() }

    if ! cfg!(qux="foo") { panic!() }
    if   cfg!(not(qux="foo")) { panic!() }

    if ! cfg!(all(foo, qux="foo")) { panic!() }
    if   cfg!(not(all(foo, qux="foo"))) { panic!() }
    if   cfg!(all(not(all(foo, qux="foo")))) { panic!() }

    if cfg!(not_a_cfg) { panic!() }
    if cfg!(all(not_a_cfg, foo, qux="foo")) { panic!() }
    if cfg!(all(not_a_cfg, foo, qux="foo")) { panic!() }
    if ! cfg!(any(not_a_cfg, foo)) { panic!() }

    if ! cfg!(not(not_a_cfg)) { panic!() }
    if ! cfg!(all(not(not_a_cfg), foo, qux="foo")) { panic!() }
}
