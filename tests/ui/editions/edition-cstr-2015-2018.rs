macro_rules! construct { ($x:ident) => { $x"str" } }
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `"str"`
    //~| NOTE expected one of 8 possible tokens

macro_rules! contain { () => { c"str" } }
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `"str"`
    //~| NOTE expected one of 8 possible tokens
    //~| NOTE you may be trying to write a c-string literal
    //~| NOTE c-string literals require Rust 2021 or later
    //~| HELP pass `--edition 2024` to `rustc`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide

fn check_macro_construct() {
    construct!(c); //~ NOTE in this expansion of construct!
}

fn check_macro_contain() {
    contain!();
    //~^ NOTE in this expansion of contain!
    //~| NOTE in this expansion of contain!
    //~| NOTE in this expansion of contain!
    //~| NOTE in this expansion of contain!
    //~| NOTE in this expansion of contain!
}

fn check_basic() {
    c"str";
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `"str"`
    //~| NOTE expected one of 8 possible tokens
    //~| NOTE you may be trying to write a c-string literal
    //~| NOTE c-string literals require Rust 2021 or later
    //~| HELP pass `--edition 2024` to `rustc`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn check_craw() {
    cr"str";
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `"str"`
    //~| NOTE expected one of 8 possible tokens
    //~| NOTE you may be trying to write a c-string literal
    //~| NOTE c-string literals require Rust 2021 or later
    //~| HELP pass `--edition 2024` to `rustc`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn check_craw_hash() {
    cr##"str"##;
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `#`
    //~| NOTE expected one of 8 possible tokens
    //~| NOTE you may be trying to write a c-string literal
    //~| NOTE c-string literals require Rust 2021 or later
    //~| HELP pass `--edition 2024` to `rustc`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn check_cstr_space() {
    c "str";
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `"str"`
    //~| NOTE expected one of 8 possible tokens
}

fn main() {}
