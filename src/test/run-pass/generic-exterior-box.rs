

type recbox[T] = rec(@T x);

fn reclift[T](&T t) -> recbox[T] { ret rec(x=@t); }

fn main() {
    let int foo = 17;
    let recbox[int] rbfoo = reclift[int](foo);
    assert (rbfoo.x == foo);
}