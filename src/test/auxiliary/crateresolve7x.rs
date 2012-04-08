// xfail-fast
// aux-build:crateresolve6-1.rs
// aux-build:crateresolve6-2.rs

// These both have the same version but differ in other metadata
mod a {
    use cr6_1 (name = "crateresolve6", vers = "0.1", calories="100");
    fn f() -> int { cr6_1::f() }
}

mod b {
    use cr6_2 (name = "crateresolve6", vers = "0.1", calories="200");
    fn f() -> int { cr6_2::f() }
}
