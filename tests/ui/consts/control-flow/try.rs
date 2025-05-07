// The `?` operator is still not const-evaluatable because it calls `From::from` on the error
// variant.

const fn opt() -> Option<i32> {
    let x = Some(2);
    x?;
    //~^ ERROR: `?` is not allowed
    //~| ERROR: `?` is not allowed
    None
}

fn main() {}
