// This test covers similar crashes from both #126521 and #126751.

macro_rules! foo {
    ($val:ident) => {
        true;
    };
}

macro_rules! bar {
    ($val:ident) => {
        (5_i32.overflowing_sub(3));
    };
}

fn main() {
    #[expect(semicolon_in_expressions_from_macros)]
    //~^ ERROR the `#[expect]` attribute is an experimental feature
    let _ = foo!(x);

    #[expect(semicolon_in_expressions_from_macros)]
    //~^ ERROR the `#[expect]` attribute is an experimental feature
    let _ = bar!(x);
}
