// error-pattern: not a sendable value

fn main() {
    let x = @3u;
    let _f = sendfn(y: uint) -> uint { ret *x+y; };
}