fn foo() {
    S {};
    S { x, y: 32, };
    S { x, y: 32, ..Default::default() };
    TupleStruct { 0: 1 };
}
