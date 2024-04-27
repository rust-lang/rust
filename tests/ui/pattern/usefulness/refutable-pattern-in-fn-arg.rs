fn main() {
    let f = |3: isize| println!("hello");
    //~^ ERROR refutable pattern in closure argument
    //~| `..=2_isize` and `4_isize..` not covered
    f(4);
}
