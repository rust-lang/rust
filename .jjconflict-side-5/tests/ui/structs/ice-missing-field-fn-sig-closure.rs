// issue-link: https://github.com/rust-lang/rust/issues/160591
// A closure call with a struct literal missing a field shouldn't ICE when checking the fn sig.

trait Context {}
struct Wrapper<C: Context + 'static> {
    container: &'static C,
}

fn main() {
    let c = |_: Wrapper<()>| {}; //~ ERROR the trait bound `(): Context` is not satisfied
    c(Wrapper { /* missing */ });
    //~^ ERROR the trait bound `(): Context` is not satisfied
    //~^^ ERROR missing field `container` in initializer of `Wrapper<_>`
    //~^^^ ERROR the trait bound `(): Context` is not satisfied
    //~^^^^ ERROR the trait bound `(): Context` is not satisfied
}
