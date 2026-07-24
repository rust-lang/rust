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

fn main() {}
