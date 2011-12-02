// error-pattern:unresolved import

mod m1 {
    fn foo() { log "foo"; }
}

mod m2 {
    import m1::foo;
}

mod m3 {
    import m2::foo;
}

fn main() { }
