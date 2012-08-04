fn main() {
    let x = [ @[true], ..512 ];
    let y = [ 0, ..1 ];
    error!("%?", x);
    error!("%?", y);
}

