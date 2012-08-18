struct defer {
    x: &[&str];
    new(x: &[&str]) { self.x = x; }
    drop { #error["%?", self.x]; }
}

fn main() {
    let _x = defer(~["Goodbye", "world!"]); //~ ERROR illegal borrow
}
