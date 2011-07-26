

type box[T] = rec(@T c);

fn unbox[T](&box[T] b) -> T { ret *b.c; }

fn main() {
    let int foo = 17;
    let box[int] bfoo = rec(c=@foo);
    log "see what's in our box";
    assert (unbox[int](bfoo) == foo);
}