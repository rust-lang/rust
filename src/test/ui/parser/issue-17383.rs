// compile-flags: -Z parse-only

enum X {
    A =
        b'a' //~ ERROR discriminator values can only be used with a field-less enum
    ,
    B(isize)
}

fn main() {}
