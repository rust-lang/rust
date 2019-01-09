fn force<F>(f: F) where F: FnOnce() { f(); }
fn main() {
    let x: isize;
    force(|| {  //~ ERROR capture of possibly uninitialized variable: `x`
        println!("{}", x);
    });
}
