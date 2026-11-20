// #106728

fn main() {
    for i in 0.2 {
        //~^ ERROR `{float}` is not an iterator
        //~| NOTE `{float}` is not an iterator
        //~| NOTE in this expansion of desugaring of `for` loop
        //~| NOTE in this expansion of desugaring of `for` loop
        //~| NOTE in this expansion of desugaring of `for` loop
        //~| NOTE in this expansion of desugaring of `for` loop
        //~| NOTE if you want to iterate between `start` until a value `end`, use the exclusive range syntax `start..end` or the inclusive range syntax `start..=end`
        //~| NOTE required for `{float}` to implement `IntoIterator`
        println!();
    }
}
