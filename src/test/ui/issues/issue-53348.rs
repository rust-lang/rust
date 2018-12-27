fn main() {
    let mut v = vec!["hello", "this", "is", "a", "test"];

    let v2 = Vec::new();

    v.into_iter().map(|s|s.to_owned()).collect::<Vec<_>>();

    let mut a = String::new();
    for i in v {
        a = *i.to_string();
        //~^ ERROR mismatched types
        //~| NOTE expected struct `std::string::String`, found str
        //~| NOTE expected type
        v2.push(a);
    }
}
