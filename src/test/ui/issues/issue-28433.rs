// compile-flags: -Z continue-parse-after-error

enum Bird {
    pub Duck,
    //~^ ERROR expected identifier, found keyword `pub`
    //~| ERROR missing comma
    //~| WARN variant `pub` should have an upper camel case name
    Goose
}


fn main() {
    let y = Bird::Goose;
}
