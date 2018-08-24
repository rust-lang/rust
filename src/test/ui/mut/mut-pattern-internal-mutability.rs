// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let foo = &mut 1;

    let &mut x = foo;
    x += 1; //[ast]~ ERROR cannot assign twice to immutable variable
            //[mir]~^ ERROR cannot assign twice to immutable variable `x`

    // explicitly mut-ify internals
    let &mut mut x = foo;
    x += 1;

    // check borrowing is detected successfully
    let &mut ref x = foo;
    *foo += 1; //[ast]~ ERROR cannot assign to `*foo` because it is borrowed
    //[mir]~^ ERROR cannot assign to `*foo` because it is borrowed
    drop(x);
}
