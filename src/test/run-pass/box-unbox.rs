

type box<T: Copy> = {c: @T};

fn unbox<T: Copy>(b: box<T>) -> T { return *b.c; }

fn main() {
    let foo: int = 17;
    let bfoo: box<int> = {c: @foo};
    debug!("see what's in our box");
    assert (unbox::<int>(bfoo) == foo);
}
