// Test that we don't ICE after trying to construct a cross-file snippet #63800.

//@ compile-flags: --test

#[macro_use]
#[path = "move-error-snippets-ext.rs"]
mod move_error_snippets_ext;

struct A;

macro_rules! sss {
    () => {
        #[test]
        fn fff() {
            static D: A = A;
            aaa!(D);         //~ ERROR cannot move
        }
    };
}

sss!();

fn main() {}
