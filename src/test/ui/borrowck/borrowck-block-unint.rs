fn force<F>(f: F) where F: FnOnce() { f(); }
fn main() {
    let x: isize;
    force(|| {  //~ ERROR borrow of possibly uninitialized variable: `x`
        println!("{}", x);
    });
}
