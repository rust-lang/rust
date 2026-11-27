//@ run-pass
// Check that least upper bound coercions don't resolve type variable merely based on the first
// coercion. Check issue #136420.

fn foo() {}
fn bar() {}

fn infer<T>(_: T) {}

fn infer_array_element<T>(_: [T; 2]) {}

fn main() {
    // Previously the element type's ty var will be unified with `foo`.
    let _: [_; 2] = [foo, bar];
    infer_array_element([foo, bar]);

    let _ = if false {
        foo
    } else {
        bar
    };
    infer(if false {
        foo
    } else {
        bar
    });

    let _ = match false {
        true => foo,
        false => bar,
    };
    infer(match false {
        true => foo,
        false => bar,
    });
}
