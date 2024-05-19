// Related to #57994.
use std::pin::Pin;
struct S;

impl S {
    fn x(self: Pin<&mut Self>) {} //~ NOTE method is available for `Pin<&mut S>`
    fn y(self: Pin<&Self>) {} //~ NOTE method is available for `Pin<&S>`
}

fn main() {
    Pin::new(&S).x(); //~ ERROR no method named `x` found for struct `Pin<&S>` in the current scope
    Pin::new(&mut S).y(); //~ ERROR no method named `y` found for struct `Pin<&mut S>` in the current scope
}
