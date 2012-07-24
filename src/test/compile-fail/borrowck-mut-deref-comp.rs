enum foo = ~int;

fn borrow(x: @mut foo) {
    let _y = &***x; //~ ERROR illegal borrow unless pure: creating immutable alias to aliasable, mutable memory
    *x = foo(~4); //~ NOTE impure due to assigning to dereference of mutable @ pointer
}

fn main() {
}