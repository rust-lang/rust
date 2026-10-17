#![feature(unboxed_closures)]
#![feature(fn_traits)]

fn main() {
    let handlers: Option<Box<dyn for<'a> FnMut<&'a mut (), Output=()>>> = None;
    handlers.unwrap().as_mut().call_mut(&mut ()); //~ ERROR: `&mut ()` is not a tuple
}
