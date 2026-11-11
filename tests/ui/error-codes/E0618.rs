enum X {
    Entry,
}

fn main() {
    X::Entry();
    //~^ ERROR expected function, found enum variant `X::Entry` [E0618]
    let x = 0i32;
    x();
    //~^ ERROR expected function, found `i32` [E0618]
}
