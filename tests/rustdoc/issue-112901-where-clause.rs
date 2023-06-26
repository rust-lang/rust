#![crate_name = "foo"]
#![feature(trivial_bounds)]

pub mod structs {
    // @has foo/structs/struct.A.html
    // @snapshot structs-A - '//pre[@class="rust item-decl"]'
    pub struct A<T>(T)
    where
        T: Copy;

    // @has foo/structs/struct.S.html
    // @snapshot structs-S - '//pre[@class="rust item-decl"]'
    pub struct S
    where
        String: Clone;
}

pub mod assoc {
    // @has foo/assoc/struct.S.html
    // @snapshot assoc-S-impl-F - '//section[@id="associatedtype.F"]'
    pub struct S;

    // @has foo/assoc/trait.Tr.html
    // @snapshot assoc-Tr-decl - '//pre[@class="rust item-decl"]'
    // @snapshot assoc-Tr-F - '//section[@id="associatedtype.F"]'
    // @snapshot assoc-Tr-impl-F - '//section[@id="associatedtype.F-1"]'
    pub trait Tr {
        type F<T>
        where
            T: Clone;
    }

    impl Tr for S {
        type F<T> = T where T: Clone;
    }
}
