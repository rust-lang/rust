// error-pattern: not a sendable value

fn main() {
    let x = @3u;
    let _f = fn~(y: uint) -> uint { ret *x+y; };
}