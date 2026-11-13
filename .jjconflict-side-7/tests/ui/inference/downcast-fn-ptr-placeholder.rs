//! Regression test for <https://github.com/rust-lang/rust/issues/23041>.
//! Previously ICEd with cat_expr error, fixed by delaying bug.

use std::any::Any;
fn main()
{
    fn bar(x:i32) ->i32 { 3*x };
    let b:Box<dyn Any> = Box::new(bar as fn(_)->_);
    b.downcast_ref::<fn(_)->_>(); //~ ERROR E0282
}
