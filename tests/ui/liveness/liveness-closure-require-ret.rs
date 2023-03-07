fn force<F>(f: F) -> isize where F: FnOnce() -> isize { f() }
fn main() { println!("{}", force(|| {})); } //~ ERROR mismatched types
