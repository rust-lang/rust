//! Deferred method calls must still report errors when later uses do not constrain their receivers.

struct Value;

impl Value {
    fn get(self) -> i32 {
        0
    }
}

fn unconstrained_index() {
    let array: [i64; 1] = [0];
    let _get = |index| array[index].pow(1);
    //~^ ERROR type annotations needed
}

fn unconstrained_receiver() {
    let get = |value| value.get();
    //~^ ERROR type annotations needed

    let _: i32 = get(Value);
}

fn nested_closure_param_ty_var() {
    let get = |value: &_| (*value).get();
    //~^ ERROR type annotations needed

    let _: i32 = get(&Value);
}

// Upstream `main` performs method lookup before either closure call can constrain `index`, so it
// reports E0282 at `array[index]`. With deferred method confirmation, the first call fixes `index`
// to `usize`, and the second call reaches the usual closure argument check and reports E0308.
fn conflicting_closure_call_types() {
    let array: [i64; 1] = [0];
    let get = |index| array[index].pow(1);

    let _: i64 = get(0usize);
    let _ = get(0u8);
    //~^ ERROR mismatched types
}

fn main() {}
