fn main() {
    let foo = &mut 1;

    let &mut x = foo;
    x += 1; //~ ERROR cannot assign twice to immutable variable `x`

    // explicitly mut-ify internals
    let &mut mut x = foo;
    x += 1;

    // check borrowing is detected successfully
    let &mut ref x = foo;
    *foo += 1; //~ ERROR cannot assign to `*foo` because it is borrowed
    drop(x);
}
