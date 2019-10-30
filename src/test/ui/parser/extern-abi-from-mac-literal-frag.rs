// check-pass

// In this test we check that the parser accepts an ABI string when it
// comes from a macro `literal` fragment as opposed to a hardcoded string.

fn main() {}

macro_rules! abi_from_lit_frag {
    ($abi:literal) => {
        extern $abi {
            fn _import();
        }

        extern $abi fn _export() {}

        type _PTR = extern $abi fn();
    }
}

mod rust {
    abi_from_lit_frag!("Rust");
}

mod c {
    abi_from_lit_frag!("C");
}
