#![allow(clashing_extern_declarations)]
//@ check-pass

// In this test we check that the parser accepts an ABI string when it
// comes from a macro `literal` or `expr` fragment as opposed to a hardcoded string.

fn main() {}

macro_rules! abi_from_lit_frag {
    ($abi:literal) => {
        extern $abi {
            fn _import();
        }

        unsafe extern $abi {}

        extern $abi fn _export() {}

        type _PTR = extern $abi fn();
    }
}

macro_rules! abi_from_expr_frag {
    ($abi:expr) => {
        extern $abi {
            fn _import();
        }

        unsafe extern $abi {}

        extern $abi fn _export() {}

        type _PTR = extern $abi fn();
    };
}

mod rust {
    abi_from_lit_frag!("Rust");
}

mod c {
    abi_from_lit_frag!("C");
}

mod rust_expr {
    abi_from_expr_frag!("Rust");
}

mod c_expr {
    abi_from_expr_frag!("C");
}
