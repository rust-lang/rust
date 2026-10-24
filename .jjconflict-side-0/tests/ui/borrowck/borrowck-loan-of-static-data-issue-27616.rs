use std::mem;

fn leak<T>(mut b: Box<T>) -> &'static mut T {
    // isn't this supposed to be safe?
    let inner = &mut *b as *mut _;
    mem::forget(b);
    unsafe { &mut *inner }
}

fn evil(mut s: &'static mut String)
{
    // create alias
    let alias: &'static mut String = s;
    let inner: &str = &alias;
    // free value
    *s = String::new(); //~ ERROR cannot assign
    let _spray = "0wned".to_owned();
    // ... and then use it
    println!("{}", inner);
}

fn main() {
    evil(leak(Box::new("hello".to_owned())));
}
