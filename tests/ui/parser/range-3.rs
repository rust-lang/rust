// Test range syntax - syntax errors.

pub fn main() {
    let r = 1..2..3;
    //~^ ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `..`
}
