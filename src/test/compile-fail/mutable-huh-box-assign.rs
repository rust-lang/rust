// error-pattern: assigning to immutable box

fn main() {
    fn f(&&v: @const int) {
        // This shouldn't be possible
        *v = 1
    }

    let v = @0;

    f(v);
}
