// This test had to do with an outdated version of the iterable trait.
// However, the condition it was testing seemed complex enough to
// warrant still having a test, so I inlined the old definitions.

trait iterable<A> {
    fn iter(blk: fn(A));
}

impl<A> fn@(fn(A)): iterable<A> {
    fn iter(blk: fn(A)) { self(blk); }
}

impl fn@(fn(uint)): iterable<uint> {
    fn iter(blk: fn(&&uint)) { self( |i| blk(i) ) }
}

fn filter<A,IA:iterable<A>>(self: IA, prd: fn@(A) -> bool, blk: fn(A)) {
    do self.iter |a| {
        if prd(a) { blk(a) }
    }
}

fn foldl<A,B,IA:iterable<A>>(self: IA, +b0: B, blk: fn(B, A) -> B) -> B {
    let mut b <- b0;
    do self.iter |a| {
        b <- blk(b, a);
    }
    return b;
}

fn range(lo: uint, hi: uint, it: fn(uint)) {
    let mut i = lo;
    while i < hi {
        it(i);
        i += 1u;
    }
}

fn main() {
    let range = |a| range(0u, 1000u, a);
    let filt = |a| filter(
        range,
        |&&n: uint| n % 3u != 0u && n % 5u != 0u,
        a);
    let sum = foldl(filt, 0u, |accum, &&n: uint| accum + n );

    io::println(fmt!{"%u", sum});
}