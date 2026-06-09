// Test that correct syntax is used in suggestion to constrain associated type

trait X {
    type Y<T>;
}

fn f<T: X>(a: T::Y<i32>) {
    //~^ HELP consider constraining the associated type `<T as X>::Y<i32>` to `Vec<i32>`
    //~| SUGGESTION Y<i32> = Vec<i32>>
    let b: Vec<i32> = a;
    //~^ ERROR mismatched types
}

fn main() {}
