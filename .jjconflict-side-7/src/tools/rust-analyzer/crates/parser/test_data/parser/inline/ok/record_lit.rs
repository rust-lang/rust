fn foo() {
    S {};
    S { x };
    S { x, y: 32, };
    S { x, y: 32, ..Default::default() };
    S { x, y: 32, .. };
    S { .. };
    S { x: ::default() };
    TupleStruct { 0: 1 };
}
