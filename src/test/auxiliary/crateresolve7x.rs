// xfail-fast
// aux-build:crateresolve_calories-1.rs
// aux-build:crateresolve_calories-2.rs

// These both have the same version but differ in other metadata
mod a {
    #[legacy_exports];
    extern mod cr_1 (name = "crateresolve_calories", vers = "0.1", calories="100");
    fn f() -> int { cr_1::f() }
}

mod b {
    #[legacy_exports];
    extern mod cr_2 (name = "crateresolve_calories", vers = "0.1", calories="200");
    fn f() -> int { cr_2::f() }
}
