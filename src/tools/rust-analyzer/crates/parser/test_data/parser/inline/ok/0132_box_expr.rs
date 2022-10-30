fn foo() {
    let x = box 1i32;
    let y = (box 1i32, box 2i32);
    let z = Foo(box 1i32, box 2i32);
}
