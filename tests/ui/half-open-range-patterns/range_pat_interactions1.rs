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
            errors_only.push(x);
        } else if let 2..3 | 4 = x {
            if_lettable.push(x);
        }
        match x as i32 {
            0..5+1 => errors_only.push(x),
            //~^ error: expected a pattern range bound, found an expression
            1 | -3..0 => first_or.push(x),
            y @ (0..5 | 6) => or_two.push(y),
            y @ -5.. => range_from.push(y),
            y @ ..-7 => assert_eq!(y, -8),
            y => bottom.push(y),
        }
    }
}
