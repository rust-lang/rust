enum SimpleEnum {
    NoState,
}

struct SimpleStruct {
    no_state_here: u64,
}

fn main() {
    let _ = |simple| {
        match simple {
            SimpleStruct {
                state: 0,
                //~^ struct `SimpleStruct` does not have a field named `state` [E0026]
                ..
            } => (),
        }
    };

    let _ = |simple| {
        match simple {
            SimpleStruct {
                no_state_here: 0,
                no_state_here: 1
                //~^ ERROR field `no_state_here` bound multiple times in the pattern [E0025]
            } => (),
        }
    };

    let _ = |simple| {
        match simple {
            SimpleEnum::NoState {
                state: 0
                //~^ ERROR variant `SimpleEnum::NoState` does not have a field named `state` [E0026]
            } => (),
        }
    };
}
