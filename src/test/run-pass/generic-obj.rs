

obj buf[T](tup(T, T, T) data) {
    fn get(int i) -> T {
        if (i == 0) {
            ret data._0;
        } else { if (i == 1) { ret data._1; } else { ret data._2; } }
    }
    fn take(&T t) { }
    fn take2(&T t) { }
}

fn main() {
    let buf[int] b = buf[int](tup(1, 2, 3));
    log "constructed object";
    log b.get(0);
    log b.get(1);
    log b.get(2);
    assert (b.get(0) == 1);
    assert (b.get(1) == 2);
    assert (b.get(2) == 3);
    b.take2(0);
}