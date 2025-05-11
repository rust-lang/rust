#![warn(clippy::crate_in_macro_def)]

mod hygienic {
    #[macro_export]
    macro_rules! print_message_hygienic {
        () => {
            println!("{}", $crate::hygienic::MESSAGE);
        };
    }

    pub const MESSAGE: &str = "Hello!";
}

mod unhygienic {
    #[macro_export]
    macro_rules! print_message_unhygienic {
        () => {
            println!("{}", crate::unhygienic::MESSAGE);
            //~^ crate_in_macro_def
        };
    }

    pub const MESSAGE: &str = "Hello!";
}

mod unhygienic_intentionally {
    // For cases where the use of `crate` is intentional, applying `allow` to the macro definition
    // should suppress the lint.
    #[allow(clippy::crate_in_macro_def)]
    #[macro_export]
    macro_rules! print_message_unhygienic_intentionally {
        () => {
            println!("{}", crate::CALLER_PROVIDED_MESSAGE);
        };
    }
}

#[macro_use]
mod not_exported {
    macro_rules! print_message_not_exported {
        () => {
            println!("{}", crate::not_exported::MESSAGE);
        };
    }

    pub const MESSAGE: &str = "Hello!";
}

fn main() {
    print_message_hygienic!();
    print_message_unhygienic!();
    print_message_unhygienic_intentionally!();
    print_message_not_exported!();
}

pub const CALLER_PROVIDED_MESSAGE: &str = "Hello!";
