fn main() {
    let f = |3: isize| println!("hello");
    //~^ ERROR refutable pattern in closure argument
    //~| NOTE `..=2_isize` and `4_isize..` not covered
    //~| NOTE the matched value is of type `isize`
    f(4);
}
