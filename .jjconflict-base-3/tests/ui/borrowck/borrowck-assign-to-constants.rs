static foo: isize = 5;

fn main() {
    // assigning to various global constants
    foo = 6; //~ ERROR cannot assign to immutable static item `foo`
}
