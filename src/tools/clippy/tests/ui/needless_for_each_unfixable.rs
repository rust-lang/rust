//@no-rustfix: overlapping suggestions
#![warn(clippy::needless_for_each)]
#![allow(clippy::needless_return)]

fn main() {
    let v: Vec<i32> = Vec::new();
    // This is unfixable because the closure includes `return`.
    v.iter().for_each(|v| {
        //~^ needless_for_each

        if *v == 10 {
            return;
        } else {
            println!("{v}");
        }
    });
}

fn issue9912() {
    let mut i = 0;
    // Changing this to a `for` loop would break type inference
    [].iter().for_each(move |_: &i32| {
        //~^ needless_for_each
        i += 1;
    });

    // Changing this would actually be okay, but we still suggest `MaybeIncorrect`ly
    [1i32].iter().for_each(move |_: &i32| {
        //~^ needless_for_each
        i += 1;
    });
}
