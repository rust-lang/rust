// run-pass

// aux-build:xcrate.rs

extern crate xcrate;

fn main() {
//  NOTE line below commeted out due to issue #45994
//  assert_eq!(xcrate::fourway_add(1)(2)(3)(4), 10);
    xcrate::return_closure_accessing_internal_fn()();
}
