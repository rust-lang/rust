

type box[T] = tup(@T);

fn unbox[T](&box[T] b) -> T { ret *b._0; }

fn main() {
    let int foo = 17;
    let box[int] bfoo = tup(@foo);
    log "see what's in our box";
    assert (unbox[int](bfoo) == foo);
}