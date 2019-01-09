// This file must never have a trailing newline
//
// revisions: ast mir
// compile-flags: -Z borrowck=compare

fn id<T>(x: T) -> T { x }

fn main() {
    let x = Some(3);
    let y = x.as_ref().unwrap_or(&id(5));
    //[ast]~^ ERROR borrowed value does not live long enough (Ast)
    //[mir]~^^ ERROR borrowed value does not live long enough (Ast)
    // This actually passes in mir
}
