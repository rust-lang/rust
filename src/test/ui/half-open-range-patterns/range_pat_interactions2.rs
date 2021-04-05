fn main() {
    let mut first_or = Vec::<i32>::new();
    let mut or_two = Vec::<i32>::new();
    let mut range_from = Vec::<i32>::new();
    let mut bottom = Vec::<i32>::new();
    let mut errors_only = Vec::<i32>::new();

    for x in -9 + 1..=(9 - 2) {
        match x as i32 {
            0..=(5+1) => errors_only.push(x),
            //~^ error: inclusive range with no end
            //~| error: expected one of `=>`, `if`, or `|`, found `(`
            1 | -3..0 => first_or.push(x),
            y @ (0..5 | 6) => or_two.push(y),
            y @ 0..const { 5 + 1 } => assert_eq!(y, 5),
            y @ -5.. => range_from.push(y),
            y @ ..-7 => assert_eq!(y, -8),
            y => bottom.push(y),
        }
    }
}
