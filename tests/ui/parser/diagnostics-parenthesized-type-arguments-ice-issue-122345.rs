//@ dont-require-annotations: NOTE

fn main() {
    unsafe {
        dealloc(ptr2, Layout::(x: !)(1, 1)); //~ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
        //~^ ERROR cannot find function `dealloc` in this scope [E0425]
        //~| ERROR cannot find value `ptr2` in this scope [E0425]
        //~| ERROR cannot find function, tuple struct or tuple variant `Layout` in this scope [E0425]
    }
}
