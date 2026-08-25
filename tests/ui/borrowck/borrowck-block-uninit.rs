fn force<F>(f: F) where F: FnOnce() { f(); }
fn main() {
    let x: isize;
    force(|| {  //~ ERROR E0381
        println!("{}", x);
    });
}
