


// -*- rust -*-
fn id[T](&T x) -> T { ret x; }

type triple = tup(int, int, int);

fn main() {
    auto x = 62;
    auto y = 63;
    auto a = 'a';
    auto b = 'b';
    let triple p = tup(65, 66, 67);
    let triple q = tup(68, 69, 70);
    y = id[int](x);
    log y;
    assert (x == y);
    b = id[char](a);
    log b;
    assert (a == b);
    q = id[triple](p);
    x = p._2;
    y = q._2;
    log y;
    assert (x == y);
}