fn main() {
    fn f(&&v: {const field: int}) {
        // This shouldn't be possible
        v.field = 1 //! ERROR assigning to const field
    }

    let v = {field: 0};

    f(v);
}
