fn a() -> i32 {
    3
}

pub fn main() {
    assert_eq!(a, 0);
    //~^ ERROR can't compare
    //~| ERROR mismatched types
    //~| ERROR doesn't implement
}
