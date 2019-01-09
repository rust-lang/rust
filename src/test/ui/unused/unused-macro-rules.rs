#![deny(unused_macros)]

// Most simple case
macro_rules! unused { //~ ERROR: unused macro definition
    () => {};
}

// Test macros created by macros
macro_rules! create_macro {
    () => {
        macro_rules! m { //~ ERROR: unused macro definition
            () => {};
        }
    };
}
create_macro!();

#[allow(unused_macros)]
mod bar {
    // Test that putting the #[deny] close to the macro's definition
    // works.

    #[deny(unused_macros)]
    macro_rules! unused { //~ ERROR: unused macro definition
        () => {};
    }
}

fn main() {}
