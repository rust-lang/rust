//@ dont-require-annotations: NOTE

const fn foo() -> ! {
    unsafe { std::mem::transmute(()) } //~ NOTE inside `foo`
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
//~^ ERROR value of the never type

const BAR: [empty::Empty; 3] = [unsafe { std::mem::transmute(()) }; 3];
//~^ ERROR value of uninhabited type

fn main() {
    FOO;
    BAR;
}
