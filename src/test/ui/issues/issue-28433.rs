// compile-flags: -Z continue-parse-after-error

enum bird {
    pub duck,
    //~^ ERROR: expected identifier, found keyword `pub`
    //~| ERROR: expected
    goose
}


fn main() {
    let y = bird::goose;
}
