// check-pass

// From https://github.com/rust-lang/rust/issues/54121/
//
// Whether the code compiled depended on the order of the trait bounds in
// `type T: Tr<u8, u8> + Tr<u16, u16>`
// But both should compile as order shouldn't matter.

trait Tr<A, B> {
    fn exec(a: A, b: B);
}

trait P {
    // This compiled successfully
    type T: Tr<u16, u16> + Tr<u8, u8>;
}

trait Q {
    // This didn't compile
    type T: Tr<u8, u8> + Tr<u16, u16>;
}

#[allow(dead_code)]
fn f<S: P>() {
    <S as P>::T::exec(0u8, 0u8)
}

#[allow(dead_code)]
fn g<S: Q>() {
    // A mismatched types error was emitted on this line.
    <S as Q>::T::exec(0u8, 0u8)
}

// Another reproduction of the same issue
trait Trait {
    type Type: Into<Self::Type1> + Into<Self::Type2> + Copy;
    type Type1;
    type Type2;
}

#[allow(dead_code)]
fn foo<T: Trait>(x: T::Type) {
    let _1: T::Type1 = x.into();
    let _2: T::Type2 = x.into();
}

fn main() { }
