// error-pattern: copying a noncopyable value

resource r(i: @mutable int) {
    *i = *i + 1;
}

fn f<T>(+i: [T], +j: [T]) {
    let k = i + j;
}

fn main() {
    let i1 = @mutable 0;
    let i2 = @mutable 1;
    let r1 <- [~r(i1)];
    let r2 <- [~r(i2)];
    f(r1, r2);
    log (r2, *i1);
    log (r1, *i2);
}