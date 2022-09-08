fn main() {
    let _ = 0: i32; //~ ERROR: type ascription is experimental
    let _ = 0: i32; // (error only emitted once)
}
