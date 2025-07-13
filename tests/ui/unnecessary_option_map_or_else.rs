#![warn(clippy::unnecessary_option_map_or_else)]
#![allow(
    clippy::let_and_return,
    clippy::let_unit_value,
    clippy::unnecessary_lazy_evaluations,
    clippy::unnecessary_literal_unwrap
)]

const fn identity<T>(x: T) -> T {
    x
}

const fn double_it(x: i32) -> i32 {
    x * 2
}

fn main() {
    // Expected errors
    // Basic scenario
    let option = Some(());
    option.map_or_else(|| (), |x| x); //~ ERROR: unused "map closure" when calling

    // Type ascription
    let option = Some(());
    option.map_or_else(|| (), |x: ()| x); //~ ERROR: unused "map closure" when calling

    // Auto-deref
    let string = String::new();
    let option = Some(&string);
    let _: &str = option.map_or_else(|| &string, |x| x); //~ ERROR: unused "map closure" when calling

    // Temporary variable
    let option = Some(());
    option.map_or_else(
        //~^ ERROR: unused "map closure" when calling
        || (),
        |x| {
            let tmp = x;
            tmp
        },
    );

    // Identity
    let string = String::new();
    let option = Some(&string);
    let _: &str = option.map_or_else(|| &string, identity); //~ ERROR: unused "map closure" when calling

    // Closure bound to a variable
    let do_nothing = |x: String| x;
    let string = String::new();
    let option = Some(string.clone());
    let _: String = option.map_or_else(|| string, do_nothing); //~ ERROR: unused "map closure" when calling

    // Correct usages
    let option = Some(());
    option.map_or_else(|| (), |_| ());

    let option = Some(());
    option.map_or_else(|| (), |_: ()| ());

    let string = String::new();
    let option = Some(&string);
    let _: &str = option.map_or_else(|| &string, |_| &string);

    let option = Some(());
    option.map_or_else(
        || (),
        |_| {
            let tmp = ();
            tmp
        },
    );

    let num = 5;
    let option = Some(num);
    let _: i32 = option.map_or_else(|| 0, double_it);

    let increase = |x: i32| x + 1;
    let num = 5;
    let option = Some(num);
    let _: i32 = option.map_or_else(|| 0, increase);
}
