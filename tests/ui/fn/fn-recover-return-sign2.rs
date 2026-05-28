// Separate test file because `Fn() => bool` isn't getting fixed and rustfix complained that
// even though a fix was applied the code was still incorrect

fn foo() => impl Fn() => bool {
    //~^ ERROR return types are denoted using `->`
    //~| ERROR expected one of `+`, `->`, `::`, `where`, or `{`, found `=>`
    unimplemented!()
}
