#![deny(nonstandard_style)]
#![allow(dead_code)]

fn CamelCase() {} //~ ERROR should have a snake

#[allow(nonstandard_style)]
mod test {
    fn CamelCase() {}

    #[forbid(nonstandard_style)]
    mod bad {
        fn CamelCase() {} //~ ERROR should have a snake

        static bad: isize = 1; //~ ERROR should have an upper
    }

    mod warn {
        #![warn(nonstandard_style)]

        fn CamelCase() {} //~ WARN should have a snake

        struct snake_case; //~ WARN should have an upper camel
    }
}

fn main() {}
