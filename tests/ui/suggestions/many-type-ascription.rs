fn main() {
    let _ = 0: i32; //~ ERROR: expected one of
    let _ = 0: i32; // (error only emitted once)
}
