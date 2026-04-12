macro_rules! bar (
    () => ()
);

macro_rules! foo (
    () => (
        #[allow_internal_unstable()]
        //~^ ERROR allow_internal_unstable side-steps
        //~| ERROR `#[allow_internal_unstable]` attribute cannot be used on macro calls
        bar!();
    );
);

foo!();
fn main() {}
