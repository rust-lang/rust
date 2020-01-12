fn main() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR comparison operators cannot be chained
    Vec<i32>::new();
    //~^ ERROR comparison operators cannot be chained
    (0..13).collect<Vec<i32>();
    //~^ ERROR comparison operators cannot be chained
}
