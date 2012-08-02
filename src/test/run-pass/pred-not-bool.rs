// this checks that a pred with a non-bool return
// type is rejected, even if the pred is never used

pure fn bad(a: int) -> int { return 37; } //~ ERROR Non-boolean return type

fn main() { }
