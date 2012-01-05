

type box<T: copy> = {c: @T};

fn unbox<T: copy>(b: box<T>) -> T { ret *b.c; }

fn main() {
    let foo: int = 17;
    let bfoo: box<int> = {c: @foo};
    #debug("see what's in our box");
    assert (unbox::<int>(bfoo) == foo);
}
