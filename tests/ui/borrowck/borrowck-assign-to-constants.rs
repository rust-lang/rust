static FOO: isize = 5;

fn main() {
    // assigning to various global constants
    FOO = 6; //~ ERROR cannot assign to immutable static item `FOO`
}
