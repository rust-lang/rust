


// -*- rust -*-
fn id[T](&T x) -> T { ret x; }

type triple = rec(int x, int y, int z);

fn main() {
    auto x = 62;
    auto y = 63;
    auto a = 'a';
    auto b = 'b';
    let triple p = rec(x=65, y=66, z=67);
    let triple q = rec(x=68, y=69, z=70);
    y = id[int](x);
    log y;
    assert (x == y);
    b = id[char](a);
    log b;
    assert (a == b);
    q = id[triple](p);
    x = p.z;
    y = q.z;
    log y;
    assert (x == y);
}