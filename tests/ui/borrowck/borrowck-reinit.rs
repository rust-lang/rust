fn main() {
    let mut x = Box::new(0);
    let _u = x; // error shouldn't note this move
    x = Box::new(1);
    drop(x);
    let _ = (1,x); //~ ERROR use of moved value: `x`
}
