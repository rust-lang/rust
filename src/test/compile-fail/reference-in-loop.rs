// error-pattern: overwriting x will invalidate reference y

fn main() {
    let x = [];
    let &y = x;
    while true {
        log_err y;
        x = [1];
    }
}
