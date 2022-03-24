#![warn(clippy::crate_in_macro_def)]

#[macro_use]
mod intentional {
    // For cases where use of `crate` is intentional, applying `allow` to the macro definition
    // should suppress the lint.
    #[allow(clippy::crate_in_macro_def)]
    macro_rules! print_message {
        () => {
            println!("{}", crate::CALLER_PROVIDED_MESSAGE);
        };
    }
}

fn main() {
    print_message!();
}

pub const CALLER_PROVIDED_MESSAGE: &str = "Hello!";
