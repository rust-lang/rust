#![warn(clippy::obfuscated_if_else)]
#![allow(clippy::unnecessary_lazy_evaluations, clippy::unit_arg, clippy::unused_unit)]

fn main() {
    true.then_some("a").unwrap_or("b");
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`
    true.then(|| "a").unwrap_or("b");
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`

    let a = 1;
    (a == 1).then_some("a").unwrap_or("b");
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`
    (a == 1).then(|| "a").unwrap_or("b");
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`

    let partial = (a == 1).then_some("a");
    partial.unwrap_or("b"); // not lint

    let mut a = 0;
    true.then_some(a += 1).unwrap_or(());
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`
    true.then_some(()).unwrap_or(a += 2);
    //~^ ERROR: this method chain can be written more clearly with `if .. else ..`
}
