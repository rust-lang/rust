#![warn(clippy::unnecessary_option_map_or_else)]
#![allow(
    clippy::let_and_return,
    clippy::let_unit_value,
    clippy::unnecessary_lazy_evaluations,
    clippy::unnecessary_literal_unwrap,
    clippy::needless_return
)]

const fn double_it(x: i32) -> i32 {
    x * 2
}

fn main() {
    // Expected errors
    // Basic scenario
    let option = Some(());
    option.map_or_else(|| (), |x| x); //~ unnecessary_option_map_or_else

    // Type ascription
    let option = Some(());
    option.map_or_else(|| (), |x: ()| x); //~ unnecessary_option_map_or_else

    // Auto-deref
    let string = String::new();
    let option = Some(&string);
    let _: &str = option.map_or_else(|| &string, |x| x);
    // This should in theory lint with a smarter check

    // Temporary variable
    let option = Some(());
    option.map_or_else(
        //~^ unnecessary_option_map_or_else
        || (),
        |x| {
            let tmp = x;
            tmp
        },
    );

    // Temporary variable with pattern
    let option = Some(((), ()));
    option.map_or_else(
        //~^ unnecessary_option_map_or_else
        || ((), ()),
        |x| {
            let tmp = x;
            let (a, b) = tmp;
            (a, b)
        },
    );

    // std::convert::identity
    let string = String::new();
    let option = Some(&string);
    let _: &str = option.map_or_else(|| &string, std::convert::identity); //~ unnecessary_option_map_or_else

    let x: Option<((), ())> = Some(((), ()));
    x.map_or_else(|| ((), ()), |(a, b)| (a, b));
    //~^ unnecessary_option_map_or_else

    // Returned temporary variable.
    let x: Option<()> = Some(());
    x.map_or_else(
        //~^ unnecessary_option_map_or_else
        || (),
        |n| {
            let tmp = n;
            let tmp2 = tmp;
            return tmp2;
        },
    );

    // Returned temporary variable with pattern
    let x: Option<((), ())> = Some(((), ()));
    x.map_or_else(
        //~^ unnecessary_option_map_or_else
        || ((), ()),
        |n| {
            let tmp = n;
            let (a, b) = tmp;
            return (a, b);
        },
    );

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
