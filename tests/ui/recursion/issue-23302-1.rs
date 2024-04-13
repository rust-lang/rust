// Check that an enum with recursion in the discriminant throws
// the appropriate error (rather than, say, blowing the stack).
enum X {
    A = X::A as isize, //~ ERROR E0391
}

fn main() { }
