// compile-flags: -Z borrowck=compare

fn foo() -> &'static u32 {
    let x = 0;
    &x
    //~^ ERROR `x` does not live long enough (Ast) [E0597]
    //~| ERROR cannot return reference to local variable `x` (Mir) [E0515]
}

fn main() { }
