fn main() {
    // Note: explicit type annot is required here
    // because otherwise the inference gets smart
    // and assigns a type of [mut [const int]].
    let v: [mut [mut int]] = [mut [mut 0]];

    fn f(&&v: [mut [const int]]) {
        v[0] = [3]
    }

    f(v); //! ERROR (values differ in mutability)
}
