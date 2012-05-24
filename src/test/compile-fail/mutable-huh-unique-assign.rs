fn main() {
    fn f(&&v: ~const int) {
        *v = 1 //! ERROR assigning to dereference of const ~ pointer
    }

    let v = ~0;

    f(v);
}
