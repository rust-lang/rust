use std::fmt::Debug;

fn elided(x: &i32) -> impl Copy { x }
//~^ ERROR: captures lifetime that does not appear in bounds

fn explicit<'a>(x: &'a i32) -> impl Copy { x }
//~^ ERROR: captures lifetime that does not appear in bounds

fn elided2(x: &i32) -> impl Copy + 'static { x } //~ ERROR E0759

fn explicit2<'a>(x: &'a i32) -> impl Copy + 'static { x } //~ ERROR E0759

fn foo<'a>(x: &i32) -> impl Copy + 'a { x }
//~^ ERROR explicit lifetime required in the type of `x`

fn elided3(x: &i32) -> Box<dyn Debug> { Box::new(x) } //~ ERROR E0759

fn explicit3<'a>(x: &'a i32) -> Box<dyn Debug> { Box::new(x) } //~ ERROR E0759

fn elided4(x: &i32) -> Box<dyn Debug + 'static> { Box::new(x) } //~ ERROR E0759

fn explicit4<'a>(x: &'a i32) -> Box<dyn Debug + 'static> { Box::new(x) } //~ ERROR E0759

fn elided5(x: &i32) -> (Box<dyn Debug>, impl Debug) { (Box::new(x), x) } //~ ERROR E0759

trait LifetimeTrait<'a> {}
impl<'a> LifetimeTrait<'a> for &'a i32 {}

fn with_bound<'a>(x: &'a i32) -> impl LifetimeTrait<'a> + 'static { x } //~ ERROR E0759

// Tests that a closure type containing 'b cannot be returned from a type where
// only 'a was expected.
fn move_lifetime_into_fn<'a, 'b>(x: &'a u32, y: &'b u32) -> impl Fn(&'a u32) {
    //~^ ERROR: captures lifetime that does not appear in bounds
    move |_| println!("{}", y)
}

fn ty_param_wont_outlive_static<T:Debug>(x: T) -> impl Debug + 'static {
    //~^ ERROR the parameter type `T` may not live long enough
    x
}

fn main() {}
