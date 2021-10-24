fn main() {}

extern "C" { //~ NOTE while parsing this item list starting here
    pub pub fn foo();
    //~^ ERROR expected one of `(`, `async`, `const`, `default`, `extern`, `fn`, `pub`, `unsafe`, or `use`, found keyword `pub`
    //~| NOTE expected one of 9 possible tokens
    //~| HELP there is already a visibility modifier, remove one
    //~| NOTE explicit visibility first seen here
} //~ NOTE the item list ends here
