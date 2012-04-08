// xfail-fast
// aux-build:crateresolve7-1.rs
// aux-build:crateresolve7-2.rs

// These both have the same version but differ in other metadata
mod a {
    use cr7_1 (name = "crateresolve7", vers = "0.1", calories="100");
    fn f() -> int { cr7_1::f() }
}

mod b {
    use cr7_2 (name = "crateresolve7", vers = "0.1", calories="200");
    fn f() -> int { cr7_2::f() }
}
