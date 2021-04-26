// stderr-per-bitwidth
#![feature(const_fn_transmute)]

const fn foo() -> ! {
    unsafe { std::mem::transmute(()) }
    //~^ WARN any use of this value will cause an error [const_err]
    //~| WARN the type `!` does not permit zero-initialization [invalid_value]
    //~| WARN this was previously accepted by the compiler but is being phased out
}

#[derive(Clone, Copy)]
enum Empty { }

#[warn(const_err)]
const FOO: [Empty; 3] = [foo(); 3];

#[warn(const_err)]
const BAR: [Empty; 3] = [unsafe { std::mem::transmute(()) }; 3];
//~^ ERROR it is undefined behavior to use this value
//~| WARN the type `Empty` does not permit zero-initialization

fn main() {
    FOO;
    BAR;
}
