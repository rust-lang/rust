// error-pattern: assigning to immutable field

fn main() {
    fn f(&&v: {const field: int}) {
        // This shouldn't be possible
        v.field = 1
    }

    let v = {field: 0};

    f(v);
}
