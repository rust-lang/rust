fn f() -> usize {
    let c = std::cell::RefCell::new("..");
    c.borrow().len() //~ ERROR: `c` does not live long enough
}

fn main() {
    let _ = f();
}
