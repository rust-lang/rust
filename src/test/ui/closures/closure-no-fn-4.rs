fn main() {
    let b = 2;
    let _: fn(usize) -> usize = match true {
        true => |a| a + 1,
        false => |a| a - b,
        //~^ ERROR `match` arms have incompatible types
    };
}
