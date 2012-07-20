enum foo = ~int;

fn borrow(x: @mut foo) {
    let y = &***x; //~ ERROR illegal borrow unless pure: unique value in aliasable, mutable location
    *x = foo(~4); //~ NOTE impure due to assigning to dereference of mutable @ pointer
}

fn main() {
}