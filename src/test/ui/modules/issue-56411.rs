macro_rules! import {
    ( $(($path:expr, $name:ident)),* ) => {
        $(
            #[path = $path]
            mod $name;
            pub use self::$name;
            //~^ ERROR the name `issue_56411_aux` is defined multiple times
            //~| ERROR `issue_56411_aux` is private, and cannot be re-exported

        )*
    }
}

import!(("issue-56411-aux.rs", issue_56411_aux));

fn main() {
    println!("Hello, world!");
}
