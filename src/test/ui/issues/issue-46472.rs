// compile-flags: -Z borrowck=compare

fn bar<'a>() -> &'a mut u32 {
    &mut 4
    //~^ ERROR borrowed value does not live long enough (Ast) [E0597]
    //~| ERROR cannot return reference to temporary value (Mir) [E0515]
}

fn main() { }
