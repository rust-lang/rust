

obj buf<@T>(data: {_0: T, _1: T, _2: T}) {
    fn get(i: int) -> T {
        if i == 0 {
            ret data._0;
        } else { if i == 1 { ret data._1; } else { ret data._2; } }
    }
    fn take(t: T) { }
    fn take2(t: T) { }
}

fn main() {
    let b: buf<int> = buf::<int>({_0: 1, _1: 2, _2: 3});
    log "constructed object";
    log b.get(0);
    log b.get(1);
    log b.get(2);
    assert (b.get(0) == 1);
    assert (b.get(1) == 2);
    assert (b.get(2) == 3);
    b.take2(0);
}
