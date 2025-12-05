struct Foo {
    field1: u8,
    field2: u8,
}

fn main() {
    let foo = Foo { field1: 1, field2: 2 };
    let Foo { var @ field1, .. } = foo; //~ ERROR unexpected `@` in struct pattern
    dbg!(var); //~ ERROR cannot find value `var` in this scope
    let Foo { field1: _, bar @ .. } = foo; //~ ERROR `@ ..` is not supported in struct patterns
    let Foo { bar @ .. } = foo; //~ ERROR `@ ..` is not supported in struct patterns
    let Foo { @ } = foo; //~ ERROR expected identifier, found `@`
    let Foo { @ .. } = foo; //~ ERROR expected identifier, found `@`
}
