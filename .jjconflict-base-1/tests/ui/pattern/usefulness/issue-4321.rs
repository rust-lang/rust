fn main() {
    let tup = (true, true);
    println!("foo {:}", match tup { //~ ERROR non-exhaustive patterns: `(true, false)` not covered
        (false, false) => "foo",
        (false, true) => "bar",
        (true, true) => "baz"
    });
}
