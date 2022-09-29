#![allow(unused)]
fn main() {
    let arr = &[0, 1, 2, 3];
    for _i in 0..arr.len().rev() {
        //~^ ERROR not an iterator
        //~| surround the range in parentheses
        // The above error used to say “the method `rev` exists for type `usize`”.
        // This regression test ensures it doesn't say that any more.
    }

    // Test for #102396
    for i in 1..11.rev() {
        //~^ ERROR not an iterator
        //~| HELP surround the range in parentheses
    }

    let end: usize = 10;
    for i in 1..end.rev() {
        //~^ ERROR not an iterator
        //~| HELP surround the range in parentheses
    }

    for i in 1..(end + 1).rev() {
        //~^ ERROR not an iterator
        //~| HELP surround the range in parentheses
    }

    if 1..(end + 1).is_empty() {
        //~^ ERROR not an iterator
        //~| ERROR mismatched types [E0308]
        //~| HELP surround the range in parentheses
    }

    if 1..(end + 1).is_sorted() {
        //~^ ERROR mismatched types [E0308]
        //~| ERROR `usize` is not an iterator [E0599]
        //~| HELP surround the range in parentheses
    }

    let _res: i32 = 3..6.take(2).sum();
    //~^ ERROR `{integer}` is not an iterator [E0599]
    //~| ERROR mismatched types [E0308]
    //~| HELP surround the range in parentheses

    let _sum: i32 = 3..6.sum();
    //~^ ERROR `{integer}` is not an iterator [E0599]
    //~| ERROR mismatched types [E0308]
    //~| HELP surround the range in parentheses

    let a = 1 as usize;
    let b = 10 as usize;

    for _a in a..=b.rev() {
        //~^ ERROR not an iterator
        //~| HELP surround the range in parentheses
    }

    let _res = ..10.contains(3);
    //~^ ERROR not an iterator
    //~| HELP surround the range in parentheses

    if 1..end.error_method() {
        //~^ ERROR no method named `error_method`
        //~| ERROR mismatched types [E0308]
        // Won't suggest
    }

    let _res = b.take(1)..a;
    //~^ ERROR not an iterator
}
