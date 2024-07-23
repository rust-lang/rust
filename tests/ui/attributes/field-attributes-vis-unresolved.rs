// Non-builtin attributes do not mess with field visibility resolution (issue #67006).

mod internal {
    struct S {
        #[rustfmt::skip]
        pub(in crate::internal) field: u8 // OK
    }

    struct Z(
        #[rustfmt::skip]
        pub(in crate::internal) u8 // OK
    );
}

struct S {
    #[rustfmt::skip]
    pub(in nonexistent) field: u8 //~ ERROR cannot find item `nonexistent` in this scope
}

struct Z(
    #[rustfmt::skip]
    pub(in nonexistent) u8 //~ ERROR cannot find item `nonexistent` in this scope
);

fn main() {}
