//@ known-bug: rust-lang/rust#128346

macro_rules! one_rep {
    ( $($a:ident)* ) => {
        A(
            const ${concat($a, Z)}: i32 = 3;
        )*
    };
}

fn main() {
    one_rep!(A B C);
}
