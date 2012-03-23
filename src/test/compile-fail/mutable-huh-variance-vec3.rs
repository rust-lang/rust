fn main() {
    // Note: explicit type annot is required here
    // because otherwise the inference gets smart
    // and assigns a type of [mut [const int]].
    let v: [mut[mut[int]]] = [mutable [mutable [0]]];

    fn f(&&v: [mutable [mutable [const int]]]) {
        v[0][1] = [mutable 3]
    }

    f(v); //! ERROR (values differ in mutability)
}
