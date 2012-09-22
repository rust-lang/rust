// error-pattern: unresolved

mod circ1 {
    #[legacy_exports];
    use circ1::*;
    export f1;
    export f2;
    export common;
    fn f1() { debug!("f1"); }
    fn common() -> uint { return 0u; }
}

mod circ2 {
    #[legacy_exports];
    use circ2::*;
    export f1;
    export f2;
    export common;
    fn f2() { debug!("f2"); }
    fn common() -> uint { return 1u; }
}

mod test {
    #[legacy_exports];
    use circ1::*;

    fn test() { f1066(); }
}
