fn main() {
    let unused = ();

    #![allow(unused_variables)] //~ ERROR not permitted in this context
}
