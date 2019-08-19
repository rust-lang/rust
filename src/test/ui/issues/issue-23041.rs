use std::any::Any;
fn main()
{
    fn bar(x:i32) ->i32 { 3*x };
    let b:Box<dyn Any> = Box::new(bar as fn(_)->_);
    b.downcast_ref::<fn(_)->_>(); //~ ERROR E0282
}
