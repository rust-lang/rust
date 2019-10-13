fn main() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR chained comparison
    Vec<i32>::new();
    //~^ ERROR chained comparison
    (0..13).collect<Vec<i32>();
    //~^ ERROR chained comparison
}
