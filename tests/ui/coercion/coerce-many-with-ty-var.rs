//@ run-pass
// Check that least upper bound coercions don't resolve type variable merely based on the first
// coercion. Check issue #136420.

fn foo() {}
fn bar() {}

fn infer<T>(_: T) {}

fn infer_array_element<T>(_: [T; 2]) {}

fn main() {
    infer(if false {
        foo
    } else {
        bar
    });

    infer(match false {
        true => foo,
        false => bar,
    });

    infer_array_element([foo, bar]);
}
