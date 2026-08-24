// Regression test for https://github.com/rust-lang/rust/issues/159989

fn main() {
    for<> || -> () {};
    //~^ ERROR `for<...>` binders for closures are experimental
    for<'a> || -> () |_;
    //~^ ERROR expected one of `,`, `:`, or `|`, found `;`
    //~| ERROR expected one of `,`, `:`, or `|`, found `;`
    //~| ERROR expected `{`, found `|`
    //~| ERROR `for<...>` binders for closures are experimental
}
