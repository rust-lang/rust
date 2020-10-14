fn a() => impl Fn() => bool {
    //~^ ERROR return types are denoted using `->`
    //~| ERROR expected `;` or `{`, found `=>`
    unimplemented!()
}

fn main() {
    let foo = |a: bool| => bool { a };
    dbg!(foo(false));
}
