fn take(_x: Box<isize>) {}


fn main() {

    let x: Box<isize> = Box::new(25);

    loop {
        take(x); //~ ERROR use of moved value: `x`
    }
}
