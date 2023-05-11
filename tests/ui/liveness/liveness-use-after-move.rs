fn main() {

    let x: Box<_> = 5.into();
    let y = x;

    println!("{}", *x); //~ ERROR borrow of moved value: `x`
    y.clone();
}
