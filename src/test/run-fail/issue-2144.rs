// error-pattern:ran out of stack

// Don't leak when the landing pads need to request more stack
// than is allowed during normal execution

fn useBlock(f: fn~() -> uint) { useBlock({|| 22u }) }
fn main() {
    useBlock({|| 22u });
}
