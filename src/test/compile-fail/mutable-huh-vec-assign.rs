// error-pattern: assigning to immutable vec content

fn main() {
    fn f(&&v: [mutable? int]) {
        // This shouldn't be possible
        v[0] = 1
    }

    let v = [0];

    f(v);
}
