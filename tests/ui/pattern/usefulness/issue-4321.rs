fn main() {
    let tup = (true, true);
    println!("foo {:}", match tup { //~ ERROR match is non-exhaustive
        (false, false) => "foo",
        (false, true) => "bar",
        (true, true) => "baz"
    });
}
