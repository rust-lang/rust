// Check that we do not ICE when `no_mangle` is applied to something that has no name.

#![crate_type = "lib"]
#![feature(stmt_expr_attributes)]

pub struct S([usize; 8]);

pub fn outer_function(x: S, y: S) -> usize {
    (#[no_mangle] || y.0[0])()
    //~^ ERROR `#[no_mangle]` cannot be used on a closure as it has no name
}
