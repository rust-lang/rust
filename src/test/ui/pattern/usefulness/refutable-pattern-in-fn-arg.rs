fn main() {
    let f = |3: isize| println!("hello");
    //~^ ERROR refutable pattern in function argument: `_` not covered
    f(4);
}
