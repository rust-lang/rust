#![warn(clippy::to_digit_is_some)]

fn main() {
    let c = 'x';
    let d = &c;

    let _ = d.to_digit(8).is_some();
    //~^ to_digit_is_some
    let _ = char::to_digit(c, 8).is_some();
    //~^ to_digit_is_some
}

#[clippy::msrv = "1.86"]
mod cannot_lint_in_const_context {
    fn without_const(c: char) -> bool {
        c.to_digit(8).is_some()
        //~^ to_digit_is_some
    }
    const fn with_const(c: char) -> bool {
        c.to_digit(8).is_some()
    }
}

#[clippy::msrv = "1.87"]
const fn with_const(c: char) -> bool {
    c.to_digit(8).is_some()
    //~^ to_digit_is_some
}
