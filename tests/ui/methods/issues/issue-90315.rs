#![allow(unused)]
fn main() {
    let arr = &[0, 1, 2, 3];
    for _i in 0..arr.len().rev() {
        //~^ ERROR can't call method
        //~| HELP surround the range in parentheses
        // The above error used to say “the method `rev` exists for type `usize`”.
        // This regression test ensures it doesn't say that any more.
    }

    // Test for #102396
    for i in 1..11.rev() {
        //~^ ERROR can't call method
        //~| HELP surround the range in parentheses
    }

    let end: usize = 10;
    for i in 1..end.rev() {
        //~^ ERROR can't call method
        //~| HELP surround the range in parentheses
    }

    for i in 1..(end + 1).rev() {
        //~^ ERROR can't call method
        //~| HELP surround the range in parentheses
    }

    if 1..(end + 1).is_empty() {
        //~^ ERROR can't call method
        //~| ERROR mismatched types [E0308]
        //~| HELP surround the range in parentheses
    }

    if 1..(end + 1).is_sorted() {
        //~^ ERROR mismatched types [E0308]
        //~| ERROR can't call method
        //~| HELP surround the range in parentheses
    }

    let _res: i32 = 3..6.take(2).sum();
    //~^ ERROR can't call method
    //~| ERROR mismatched types [E0308]
    //~| HELP surround the range in parentheses

    let _sum: i32 = 3..6.sum();
    //~^ ERROR can't call method
    //~| ERROR mismatched types [E0308]
    //~| HELP surround the range in parentheses

    let a = 1 as usize;
    let b = 10 as usize;

    for _a in a..=b.rev() {
        //~^ ERROR can't call method
        //~| HELP surround the range in parentheses
    }

    let _res = ..10.contains(3);
    //~^ ERROR can't call method
    //~| HELP surround the range in parentheses

    if 1..end.error_method() {
        //~^ ERROR no method named `error_method`
        //~| ERROR mismatched types [E0308]
        // Won't suggest
    }

    let _res = b.take(1)..a;
    //~^ ERROR `usize` is not an iterator

    let _res: i32 = ..6.take(2).sum();
    //~^ ERROR can't call method `take` on ambiguous numeric type
    //~| HELP you must specify a concrete type for this numeric value
    // Won't suggest because `RangeTo` dest not implemented `take`
}
