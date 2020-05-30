use std::fmt::Debug;

fn elided(x: &i32) -> impl Copy { x }
//~^ ERROR cannot infer an appropriate lifetime

fn explicit<'a>(x: &'a i32) -> impl Copy { x }
//~^ ERROR cannot infer an appropriate lifetime

fn elided2(x: &i32) -> impl Copy + 'static { x }
//~^ ERROR cannot infer an appropriate lifetime

fn explicit2<'a>(x: &'a i32) -> impl Copy + 'static { x }
//~^ ERROR cannot infer an appropriate lifetime

fn foo<'a>(x: &i32) -> impl Copy + 'a { x }
//~^ ERROR explicit lifetime required in the type of `x`

fn elided3(x: &i32) -> Box<dyn Debug> { Box::new(x) }
//~^ ERROR cannot infer an appropriate lifetime

fn explicit3<'a>(x: &'a i32) -> Box<dyn Debug> { Box::new(x) }
//~^ ERROR cannot infer an appropriate lifetime

fn elided4(x: &i32) -> Box<dyn Debug + 'static> { Box::new(x) }
//~^ ERROR cannot infer an appropriate lifetime

fn explicit4<'a>(x: &'a i32) -> Box<dyn Debug + 'static> { Box::new(x) }
//~^ ERROR cannot infer an appropriate lifetime

trait LifetimeTrait<'a> {}
impl<'a> LifetimeTrait<'a> for &'a i32 {}

fn with_bound<'a>(x: &'a i32) -> impl LifetimeTrait<'a> + 'static { x }
//~^ ERROR cannot infer an appropriate lifetime

// Tests that a closure type containing 'b cannot be returned from a type where
// only 'a was expected.
fn move_lifetime_into_fn<'a, 'b>(x: &'a u32, y: &'b u32) -> impl Fn(&'a u32) {
    //~^ ERROR lifetime mismatch
    move |_| println!("{}", y)
}

fn ty_param_wont_outlive_static<T:Debug>(x: T) -> impl Debug + 'static {
    //~^ ERROR the parameter type `T` may not live long enough
    x
}

fn main() {}
