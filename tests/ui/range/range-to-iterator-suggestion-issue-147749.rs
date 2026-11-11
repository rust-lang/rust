fn main() {
    for x in (..4).rev() {
        //~^ ERROR `RangeTo<{integer}>` is not an iterator
        //~| HELP consider using a bounded `Range` by adding a concrete starting value
        let _ = x;
    }

    for x in (..=4).rev() {
        //~^ ERROR `std::ops::RangeToInclusive<{integer}>` is not an iterator
        //~| HELP consider using a bounded `RangeInclusive` by adding a concrete starting value
        let _ = x;
    }

    // should not suggest for `iter` method
    let _v: Vec<_> = (..5).iter().collect();
    //~^ ERROR no method named `iter` found

    for _x in (..'a').rev() {}
    //~^ ERROR `RangeTo<char>` is not an iterator
    //~| HELP consider using a bounded `Range` by adding a concrete starting value

    for _x in (..='a').rev() {}
    //~^ ERROR `std::ops::RangeToInclusive<char>` is not an iterator
    //~| HELP consider using a bounded `RangeInclusive` by adding a concrete starting value

    for _x in (..-10).rev() {}
    //~^ ERROR `RangeTo<{integer}>` is not an iterator
    //~| HELP consider using a bounded `Range` by adding a concrete starting value

    for _x in (..=-10).rev() {}
    //~^ ERROR `std::ops::RangeToInclusive<{integer}>` is not an iterator
    //~| HELP consider using a bounded `RangeInclusive` by adding a concrete starting value

    let end_val = 10;
    for _x in (..-end_val).rev() {}
    //~^ ERROR `RangeTo<{integer}>` is not an iterator
    //~| HELP consider using a bounded `Range` by adding a concrete starting value
}
