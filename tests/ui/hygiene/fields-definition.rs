#![feature(decl_macro)]

macro modern($a: ident) {
    struct Modern {
        a: u8,
        $a: u8, // OK
    }
}

macro_rules! legacy {
    ($a: ident) => {
        struct Legacy {
            a: u8,
            $a: u8, //~ ERROR field `a` is already declared
        }
    }
}

modern!(a);
legacy!(a);

fn main() {}
