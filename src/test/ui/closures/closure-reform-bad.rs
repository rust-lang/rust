fn call_bare(f: fn(&str)) {
    f("Hello ");
}

fn main() {
    let string = "world!";
    let f = |s: &str| println!("{}{}", s, string);
    call_bare(f) //~ ERROR mismatched types
}
