fn main() {
    let box i = ();
    let box Outer { box i, j: box Inner(box &x) } = ();
    let box ref mut i = ();
}
