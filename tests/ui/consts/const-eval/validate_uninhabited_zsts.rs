const fn foo() -> ! {
    unsafe { std::mem::transmute(()) }
    //~^ ERROR evaluation of constant value failed
}

// Type defined in a submodule, so that it is not "visibly"
// uninhabited (which would change interpreter behavior).
pub mod empty {
    #[derive(Clone, Copy)]
    enum Void {}

    #[derive(Clone, Copy)]
    pub struct Empty(Void);
}

const FOO: [empty::Empty; 3] = [foo(); 3];

const BAR: [empty::Empty; 3] = [unsafe { std::mem::transmute(()) }; 3];
//~^ ERROR evaluation of constant value failed

fn main() {
    FOO;
    BAR;
}
