// stderr-per-bitwidth

const fn foo() -> ! {
    unsafe { std::mem::transmute(()) }
    //~^ ERROR evaluation of constant value failed
    //~| WARN the type `!` does not permit zero-initialization [invalid_value]
}

// Type defined in a submodule, so that it is not "visibly"
// uninhabited (which would change interpreter behavior).
pub mod empty {
    #[derive(Clone, Copy)]
    enum Void {}

    #[derive(Clone, Copy)]
    pub struct Empty(Void);
}

#[warn(const_err)]
const FOO: [empty::Empty; 3] = [foo(); 3];

#[warn(const_err)]
const BAR: [empty::Empty; 3] = [unsafe { std::mem::transmute(()) }; 3];
//~^ ERROR it is undefined behavior to use this value
//~| WARN the type `empty::Empty` does not permit zero-initialization

fn main() {
    FOO;
    BAR;
}
