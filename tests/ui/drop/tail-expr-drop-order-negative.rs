//@ revisions: edition2021 edition2024
//@ [edition2024] edition: 2024
//@ [edition2021] check-pass

fn why_would_you_do_this() -> bool {
    let mut x = None;
    // Make a temporary `RefCell` and put a `Ref` that borrows it in `x`.
    x.replace(std::cell::RefCell::new(123).borrow()).is_some()
    //[edition2024]~^ ERROR: temporary value dropped while borrowed
}

fn main() {
    why_would_you_do_this();
}
