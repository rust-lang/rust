// Nested lint levels should control `non_ascii_idents` at the annotated scope.

#![allow(dead_code, unused_macros)]
#![deny(non_ascii_idents, unused_attributes)]

#[allow(non_ascii_idents)]
fn föö() {}

mod allowed {
    #![allow(non_ascii_idents)]

    fn bår() {}

    macro_rules! allowed_macro_tokens {
        () => {
            let quúx = 0;
        };
    }
}

fn bår() {}
//~^ ERROR identifier contains non-ASCII characters

macro_rules! denied_macro_tokens {
    () => {
        let bazé = 0;
        //~^ ERROR identifier contains non-ASCII characters
    };
}

fn main() {}
