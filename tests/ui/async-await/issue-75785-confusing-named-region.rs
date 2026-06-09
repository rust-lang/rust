//@ edition:2018
//
// Regression test for issue #75785
// Tests that we don't point to a confusing named
// region when emitting a diagnostic

pub async fn async_fn(x: &mut i32) -> (&i32, &i32) {
    let y = &*x;
    *x += 1; //~ ERROR cannot assign to
    (&32, y)
}

fn main() {}
