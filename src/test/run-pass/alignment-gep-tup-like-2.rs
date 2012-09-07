type pair<A,B> = {
    a: A, b: B
};

enum rec<A> = _rec<A>;
type _rec<A> = {
    val: A,
    mut rec: Option<@rec<A>>
};

fn make_cycle<A:Copy>(a: A) {
    let g: @rec<A> = @rec({val: a, mut rec: None});
    g.rec = Some(g);
}

fn f<A:Send Copy, B:Send Copy>(a: A, b: B) -> fn@() -> (A, B) {
    fn@() -> (A, B) { (a, b) }
}

fn main() {
    let x = 22_u8;
    let y = 44_u64;
    let z = f(~x, y);
    make_cycle(z);
    let (a, b) = z();
    debug!("a=%u b=%u", *a as uint, b as uint);
    assert *a == x;
    assert b == y;
}