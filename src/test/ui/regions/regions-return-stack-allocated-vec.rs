// Test that we cannot return a stack allocated slice

fn function(x: isize) -> &'static [isize] {
    &[x] //~ ERROR borrowed value does not live long enough
}

fn main() {
    let x = function(1);
    let y = x[0];
}
