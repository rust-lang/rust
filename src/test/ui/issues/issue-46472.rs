// compile-flags: -Z emit-end-regions -Z borrowck=compare

fn bar<'a>() -> &'a mut u32 {
    &mut 4
    //~^ ERROR borrowed value does not live long enough (Ast) [E0597]
    //~| ERROR borrowed value does not live long enough (Mir) [E0597]
}

fn main() { }
