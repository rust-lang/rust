macro_rules! foo {
    ($val:ident) => {
        true;
    };
}

fn main() {
    #[expect(semicolon_in_expressions_from_macros)]
    //~^ ERROR the `#[expect]` attribute is an experimental feature
    let _ = foo!(x);
}
