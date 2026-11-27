#![feature(type_changing_struct_update)]

#[derive(Clone)]
struct Machine<'a, S> {
    state: S,
    lt_str: &'a str,
    common_field: i32,
}

#[derive(Clone)]
struct State1;
#[derive(Clone)]
struct State2;

fn update_to_state2() {
    let s = String::from("hello");
    let m1: Machine<State1> = Machine {
        state: State1,
        lt_str: &s,
                //~^ ERROR `s` does not live long enough [E0597]
                // FIXME: The error here actually comes from line 34. The
                // span of the error message should be corrected to line 34
        common_field: 2,
    };
    // update lifetime
    let m3: Machine<'static, State1> = Machine {
        lt_str: "hello, too",
        ..m1.clone()
    };
    // update lifetime and type
    let m4: Machine<'static, State2> = Machine {
        state: State2,
        lt_str: "hello, again",
        ..m1.clone()
    };
    // updating to `static should fail.
    let m2: Machine<'static, State1> = Machine {
        ..m1
    };
}

fn main() {}
