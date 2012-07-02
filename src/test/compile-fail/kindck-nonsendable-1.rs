fn foo(_x: @uint) {}

fn main() {
    let x = @3u;
    let _ = fn~() { foo(x); }; //~ ERROR not a sendable value
    let _ = fn~(copy x) { foo(x); }; //~ ERROR not a sendable value
    let _ = fn~(move x) { foo(x); }; //~ ERROR not a sendable value
}