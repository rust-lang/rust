

type tupbox[T] = tup(@T);

type recbox[T] = rec(@T x);

fn tuplift[T](&T t) -> tupbox[T] { ret tup(@t); }

fn reclift[T](&T t) -> recbox[T] { ret rec(x=@t); }

fn main() {
    let int foo = 17;
    let tupbox[int] tbfoo = tuplift[int](foo);
    let recbox[int] rbfoo = reclift[int](foo);
    assert (tbfoo._0 == foo);
    assert (rbfoo.x == foo);
}