//@ check-fail

const _A: = 123;
//~^ ERROR: omitting type on const item declaration is experimental [E0658]
//~| ERROR: mismatched types [E0308]

fn main() {
    const _B: = 123;
    //~^ ERROR: omitting type on const item declaration is experimental [E0658]
    //~| ERROR: mismatched types [E0308]
}
