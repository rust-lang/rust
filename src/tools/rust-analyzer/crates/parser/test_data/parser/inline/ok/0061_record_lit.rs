fn foo() {
    S {};
    S { x };
    S { x, y: 32, };
    S { x, y: 32, ..Default::default() };
    S { x: ::default() };
    TupleStruct { 0: 1 };
}
