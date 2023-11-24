fn main() {
    let f = |3: isize| println!("hello");
    //~^ ERROR refutable pattern in function argument
    //~| `..=2_isize` and `4_isize..` not covered
    f(4);
}
