fn foo() {
    let _ = f();
    let _ = f()(1)(1, 2,);
    let _ = f(<Foo>::func());
    f(<Foo as Trait>::func());
}
