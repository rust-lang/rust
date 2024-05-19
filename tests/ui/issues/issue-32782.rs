macro_rules! bar (
    () => ()
);

macro_rules! foo (
    () => (
        #[allow_internal_unstable] //~ ERROR allow_internal_unstable side-steps
        bar!();
    );
);

foo!();
fn main() {}
