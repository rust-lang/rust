fn main() {
    let mut v = vec!["hello", "this", "is", "a", "test"];

    let v2 = Vec::new();

    v.into_iter().map(|s|s.to_owned()).collect::<Vec<_>>();

    let mut a = String::new(); //~ NOTE expected due to this value
    for i in v {
        a = *i.to_string();
        //~^ ERROR mismatched types
        //~| NOTE expected struct `String`, found `str`
        v2.push(a);
    }
}
