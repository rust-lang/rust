fn main() {
    match 'a' {
        char{ch} => true
        //~^ ERROR expected struct, variant or union type, found builtin type `char`
    };
}
