// test that autoderef of a type like this does not
// cause compiler to loop.  Note that no instances
// of such a type could ever be constructed.
resource t(x: x) {} //! ERROR this type cannot be instantiated
enum x = @t; //! ERROR this type cannot be instantiated

fn new_t(x: t) {
    x.to_str; //! ERROR attempted access of field to_str
}

fn main() {
}
