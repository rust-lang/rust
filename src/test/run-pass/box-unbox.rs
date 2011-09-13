

type box<T> = {c: @T};

fn unbox<@T>(b: box<T>) -> T { ret *b.c; }

fn main() {
    let foo: int = 17;
    let bfoo: box<int> = {c: @foo};
    log "see what's in our box";
    assert (unbox::<int>(bfoo) == foo);
}
