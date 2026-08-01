//! Closure bodies still report ambiguity when their signatures receive no outside constraints.

#[derive(Copy, Clone)]
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

fn receiver_inferred_from_later_call() {
    let get = |value| value.get();

    let _: i32 = get(Value);
}

fn referenced_receiver_inferred_from_later_call() {
    let get = |value: &_| (*value).get();

    let _: i32 = get(&Value);
}

// Upstream `main` checks the closure body before either call can constrain `index`, so it reports
// E0282 at `array[index]`. With deferred body checking, the first call fixes `index` to `usize`,
// and the second call reaches the usual closure argument check and reports E0308.
fn conflicting_closure_call_types() {
    let array: [i64; 1] = [0];
    let get = |index| array[index].pow(1);

    let _: i64 = get(0usize);
    let _ = get(0u8);
    //~^ ERROR mismatched types
}

fn deferred_body_still_runs_late_checks() {
    let _repeat = |_value| [String::new(); 2];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn take_getter<F: FnOnce(Value) -> i32>(_: F) {}

fn value_argument_is_a_body_checking_boundary() {
    let get = |value| value.get();
    //~^ ERROR type annotations needed

    // Passing an existing closure as a value checks its body before this `FnOnce` bound is used.
    take_getter(get);
}

fn main() {}
