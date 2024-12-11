//@ check-pass
//@ proc-macro: test-macros.rs

#![feature(decl_macro)]
#![feature(stmt_expr_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro mac {
    (expr $expr:expr) => {
        #[derive(Print)]
        enum E {
            V = { let _ = $expr; 0 },
        }
    },
    (stmt $stmt:stmt) => {
        #[derive(Print)]
        enum E {
            V = { let _ = { $stmt }; 0 },
        }
    },
}

const PATH: u8 = 2;

fn main() {
    mac!(expr #[allow(warnings)] 0);
    mac!(stmt 0);
    mac!(stmt {});
    mac!(stmt PATH);
    mac!(stmt 0 + 1);
    mac!(stmt PATH + 1);
}
