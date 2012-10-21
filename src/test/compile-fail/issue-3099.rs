fn a(x: ~str) -> ~str {
    fmt!("First function with %s", x)
}

fn a(x: ~str, y: ~str) -> ~str { //~ ERROR duplicate definition of value a
    fmt!("Second function with %s and %s", x, y)
}

fn main() {
    info!("Result: ");
}
