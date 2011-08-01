// error-pattern:refutable pattern

tag xx {
    xx(int);
    yy;
}

fn main() {
    let @{x:xx(x), y} = @{x: xx(10), y: 20};
    assert x + y == 30;
}
