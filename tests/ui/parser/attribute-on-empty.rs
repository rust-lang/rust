// Exercise outer attributes being applied to "nothing" in invalid contexts.

struct Baz<const N: usize>(i32);

fn f() {
    let _: Baz<#[cfg(false)]> = todo!();
    //~^ ERROR attributes cannot be applied here
}

fn g(_param: #[attr]) {}
//~^ ERROR attributes cannot be applied to types
//~| ERROR expected type, found `)`

fn barrier0() {
    fn f() -> #[attr] { 0 }
    //~^ ERROR attributes cannot be applied to types
    //~| ERROR expected type, found `{`
}

struct S {
    field: #[attr],
    //~^ ERROR attributes cannot be applied to types
    //~| ERROR expected type, found `,`
}

fn barrier1() {
    type Tuple = (#[attr], String);
    //~^ ERROR attributes cannot be applied to types
    //~| ERROR expected type, found `,`
}

impl #[attr] {}
//~^ ERROR attributes cannot be applied to types
//~| ERROR expected type, found `{`

fn main() {}
