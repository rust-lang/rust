fn main() {
    fn f(&&v: [const int]) {
        // This shouldn't be possible
        v[0] = 1 //! ERROR assigning to const vec content
    }

    let v = [0];

    f(v);
}
