//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Foo<const N: u8>([u8; N as usize])
where
    [(); N as usize]:;

struct Bar<const N: u8>([u8; (N + 2) as usize]) where [(); (N + 2) as usize]:;

// unifying with subtrees
struct Evaluatable<const N: u16>;
fn foo<const N: u8>() where Evaluatable<{N as usize as u16 }>: {
    let _ = Foo::<N>([1; N as usize]);
}


fn main() {}
