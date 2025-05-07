#![warn(clippy::obfuscated_if_else)]
#![allow(
    clippy::unnecessary_lazy_evaluations,
    clippy::unit_arg,
    clippy::unused_unit,
    clippy::unwrap_or_default
)]

fn main() {
    true.then_some("a").unwrap_or("b");
    //~^ obfuscated_if_else

    true.then(|| "a").unwrap_or("b");
    //~^ obfuscated_if_else

    let a = 1;
    (a == 1).then_some("a").unwrap_or("b");
    //~^ obfuscated_if_else

    (a == 1).then(|| "a").unwrap_or("b");
    //~^ obfuscated_if_else

    let partial = (a == 1).then_some("a");
    partial.unwrap_or("b"); // not lint

    let mut a = 0;
    true.then_some(a += 1).unwrap_or(());
    //~^ obfuscated_if_else

    true.then_some(()).unwrap_or(a += 2);
    //~^ obfuscated_if_else

    let mut n = 1;
    true.then(|| n = 1).unwrap_or_else(|| n = 2);
    //~^ obfuscated_if_else
    true.then_some(1).unwrap_or_else(|| n * 2);
    //~^ obfuscated_if_else
    true.then_some(n += 1).unwrap_or_else(|| ());
    //~^ obfuscated_if_else

    let _ = true.then_some(1).unwrap_or_else(|| n * 2);
    //~^ obfuscated_if_else

    true.then_some(1).unwrap_or_else(Default::default);
    //~^ obfuscated_if_else

    let partial = true.then_some(1);
    partial.unwrap_or_else(|| n * 2); // not lint

    true.then_some(()).unwrap_or_default();
    //~^ obfuscated_if_else

    true.then(|| ()).unwrap_or_default();
    //~^ obfuscated_if_else

    true.then_some(1).unwrap_or_default();
    //~^ obfuscated_if_else

    true.then(|| 1).unwrap_or_default();
    //~^ obfuscated_if_else
}

fn issue11141() {
    // Parentheses are required around the left side of a binary expression
    let _ = true.then_some(40).unwrap_or(17) | 2;
    //~^ obfuscated_if_else

    // Parentheses are required only for the leftmost expression
    let _ = true.then_some(30).unwrap_or(17) | true.then_some(2).unwrap_or(3) | true.then_some(10).unwrap_or(1);
    //~^ obfuscated_if_else
    //~| obfuscated_if_else
    //~| obfuscated_if_else

    // Parentheses are not required around the right side of a binary expression
    let _ = 2 | true.then_some(40).unwrap_or(17);
    //~^ obfuscated_if_else

    // Parentheses are not required for a cast
    let _ = true.then_some(42).unwrap_or(17) as u8;
    //~^ obfuscated_if_else

    // Parentheses are not required for a deref
    let _ = *true.then_some(&42).unwrap_or(&17);
    //~^ obfuscated_if_else

    // Parentheses are not required for a deref followed by a cast
    let _ = *true.then_some(&42).unwrap_or(&17) as u8;
    //~^ obfuscated_if_else
}
