// The `?` operator is still not const-evaluatable because it calls `From::from` on the error
// variant.

const fn opt() -> Option<i32> {
    let x = Some(2);
    x?; //~ ERROR `?` is not allowed in a `const fn`
    //~^ ERROR: cannot convert
    //~| ERROR: cannot determine
    None
}

fn main() {}
