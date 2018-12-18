macro_rules! import {
    ( $($name:ident),* ) => {
        $(
            mod $name;
            pub use self::$name;
            //~^ ERROR the name `issue_56411` is defined multiple times
            //~| ERROR `issue_56411` is private, and cannot be re-exported

        )*
    }
}

import!(issue_56411);

fn main() {
    println!("Hello, world!");
}
