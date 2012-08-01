// error-pattern: use of moved variable

fn main() {
    let x = 3;
    let y = move x;
    debug!("%d", x);
}

