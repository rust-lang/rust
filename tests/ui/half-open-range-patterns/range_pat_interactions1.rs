fn main() {
    let mut if_lettable = Vec::<i32>::new();
    let mut first_or = Vec::<i32>::new();
    let mut or_two = Vec::<i32>::new();
    let mut range_from = Vec::<i32>::new();
    let mut bottom = Vec::<i32>::new();
    let mut errors_only = Vec::<i32>::new();

    for x in -9 + 1..=(9 - 2) {
        if let n @ 2..3|4 = x {
            //~^ error: variable `n` is not bound in all patterns
            //~| exclusive range pattern syntax is experimental
            errors_only.push(x);
        } else if let 2..3 | 4 = x {
            //~^ exclusive range pattern syntax is experimental
            if_lettable.push(x);
        }
        match x as i32 {
            0..5+1 => errors_only.push(x),
            //~^ error: expected a pattern range bound, found an expression
            //~| error: exclusive range pattern syntax is experimental
            1 | -3..0 => first_or.push(x),
            //~^ error: exclusive range pattern syntax is experimental
            y @ (0..5 | 6) => or_two.push(y),
            //~^ error: exclusive range pattern syntax is experimental
            y @ 0..const { 5 + 1 } => assert_eq!(y, 5),
            //~^ error: exclusive range pattern syntax is experimental
            //~| error: inline-const in pattern position is experimental
            y @ -5.. => range_from.push(y),
            y @ ..-7 => assert_eq!(y, -8),
            //~^ error: exclusive range pattern syntax is experimental
            y => bottom.push(y),
        }
    }
}
