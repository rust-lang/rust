// compile-flags: -Z emit-end-regions -Z borrowck=compare

fn foo() -> &'static u32 {
    let x = 0;
    &x
    //~^ ERROR `x` does not live long enough (Ast) [E0597]
    //~| ERROR `x` does not live long enough (Mir) [E0597]
}

fn main() { }
