// compile-flags: -Z continue-parse-after-error

enum Bird {
    pub Duck,
    //~^ ERROR unnecessary visibility qualifier
    Goose
}


fn main() {
    let y = Bird::Goose;
}
