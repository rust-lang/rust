#![crate_name = "qwop"]

/// (written on a spider's web) Some Macro
#[macro_export]
macro_rules! some_macro {
    () => {
        println!("this is some macro, for sure");
    };
}

/// Some other macro, to fill space.
#[macro_export]
macro_rules! other_macro {
    () => {
        println!("this is some other macro, whatev");
    };
}

/// This macro is so cool, it's Super.
#[macro_export]
macro_rules! super_macro {
    () => {
        println!("is it a bird? a plane? no, it's Super Macro!");
    };
}
