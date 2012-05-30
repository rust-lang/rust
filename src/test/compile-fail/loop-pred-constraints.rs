// https://github.com/mozilla/rust/issues/2374
// error-pattern:unsatisfied precondition constraint (for example, even(y


fn print_even(y: int) : even(y) { log(debug, y); }

pure fn even(y: int) -> bool { true }

fn main() {
    let mut y = 42;
    check (even(y));
    loop {
        print_even(y);
        loop { y += 1; break; }
    }
}
