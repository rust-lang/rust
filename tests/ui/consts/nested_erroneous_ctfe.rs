fn main() {
    [9; || [9; []]];
    //~^ ERROR: mismatched types
}
