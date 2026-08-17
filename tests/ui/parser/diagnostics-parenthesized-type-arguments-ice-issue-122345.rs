//@ dont-require-annotations: NOTE

fn main() {
    unsafe {
        dealloc(ptr2, Layout::(x: !)(1, 1)); //~ ERROR named parameters in parenthesized generic argument lists are experimental
        //~^ ERROR cannot find function `dealloc` in this scope [E0425]
        //~| ERROR cannot find value `ptr2` in this scope [E0425]
        //~| ERROR the `!` type is experimental [E0658]
        //~| ERROR cannot find function, tuple struct or tuple variant `Layout` in this scope [E0425]
    }
}
