trait base_iter<A> {
    fn each(blk: fn(A) -> bool);
    fn size_hint() -> option<uint>;
}

trait extended_iter<A> {
    fn eachi(blk: fn(uint, A) -> bool);
    fn all(blk: fn(A) -> bool) -> bool;
    fn any(blk: fn(A) -> bool) -> bool;
    fn foldl<B>(+b0: B, blk: fn(B, A) -> B) -> B;
    fn contains(x: A) -> bool;
    fn count(x: A) -> uint;
    fn position(f: fn(A) -> bool) -> option<uint>;
}

trait times {
    fn times(it: fn() -> bool);
}
trait timesi{
    fn timesi(it: fn(uint) -> bool);
}

trait copyable_iter<A:copy> {
    fn filter_to_vec(pred: fn(A) -> bool) -> ~[A];
    fn map_to_vec<B>(op: fn(A) -> B) -> ~[B];
    fn to_vec() -> ~[A];
    fn min() -> A;
    fn max() -> A;
    fn find(p: fn(A) -> bool) -> option<A>;
}

fn eachi<A,IA:base_iter<A>>(self: IA, blk: fn(uint, A) -> bool) {
    let mut i = 0u;
    for self.each |a| {
        if !blk(i, a) { break; }
        i += 1u;
    }
}

fn all<A,IA:base_iter<A>>(self: IA, blk: fn(A) -> bool) -> bool {
    for self.each |a| {
        if !blk(a) { return false; }
    }
    return true;
}

fn any<A,IA:base_iter<A>>(self: IA, blk: fn(A) -> bool) -> bool {
    for self.each |a| {
        if blk(a) { return true; }
    }
    return false;
}

fn filter_to_vec<A:copy,IA:base_iter<A>>(self: IA,
                                         prd: fn(A) -> bool) -> ~[A] {
    let mut result = ~[];
    self.size_hint().iter(|hint| vec::reserve(result, hint));
    for self.each |a| {
        if prd(a) { vec::push(result, a); }
    }
    return result;
}

fn map_to_vec<A:copy,B,IA:base_iter<A>>(self: IA, op: fn(A) -> B) -> ~[B] {
    let mut result = ~[];
    self.size_hint().iter(|hint| vec::reserve(result, hint));
    for self.each |a| {
        vec::push(result, op(a));
    }
    return result;
}

fn flat_map_to_vec<A:copy,B:copy,IA:base_iter<A>,IB:base_iter<B>>(
    self: IA, op: fn(A) -> IB) -> ~[B] {

    let mut result = ~[];
    for self.each |a| {
        for op(a).each |b| {
            vec::push(result, b);
        }
    }
    return result;
}

fn foldl<A,B,IA:base_iter<A>>(self: IA, +b0: B, blk: fn(B, A) -> B) -> B {
    let mut b <- b0;
    for self.each |a| {
        b = blk(b, a);
    }
    return b;
}

fn to_vec<A:copy,IA:base_iter<A>>(self: IA) -> ~[A] {
    foldl::<A,~[A],IA>(self, ~[], |r, a| vec::append(r, ~[a]))
}

fn contains<A,IA:base_iter<A>>(self: IA, x: A) -> bool {
    for self.each |a| {
        if a == x { return true; }
    }
    return false;
}

fn count<A,IA:base_iter<A>>(self: IA, x: A) -> uint {
    do foldl(self, 0u) |count, value| {
        if value == x {
            count + 1u
        } else {
            count
        }
    }
}

fn position<A,IA:base_iter<A>>(self: IA, f: fn(A) -> bool)
        -> option<uint> {
    let mut i = 0;
    for self.each |a| {
        if f(a) { return some(i); }
        i += 1;
    }
    return none;
}

// note: 'rposition' would only make sense to provide with a bidirectional
// iter interface, such as would provide "reach" in addition to "each". as is,
// it would have to be implemented with foldr, which is too inefficient.

fn repeat(times: uint, blk: fn() -> bool) {
    let mut i = 0u;
    while i < times {
        if !blk() { break }
        i += 1u;
    }
}

fn min<A:copy,IA:base_iter<A>>(self: IA) -> A {
    alt do foldl::<A,option<A>,IA>(self, none) |a, b| {
        alt a {
          some(a_) if a_ < b => {
            // FIXME (#2005): Not sure if this is successfully optimized to
            // a move
            a
          }
          _ => some(b)
        }
    } {
        some(val) => val,
        none => fail ~"min called on empty iterator"
    }
}

fn max<A:copy,IA:base_iter<A>>(self: IA) -> A {
    alt do foldl::<A,option<A>,IA>(self, none) |a, b| {
        alt a {
          some(a_) if a_ > b => {
            // FIXME (#2005): Not sure if this is successfully optimized to
            // a move.
            a
          }
          _ => some(b)
        }
    } {
        some(val) => val,
        none => fail ~"max called on empty iterator"
    }
}

/*
#[test]
fn test_enumerate() {
    enumerate(["0", "1", "2"]) {|i,j|
        assert fmt!{"%u",i} == j;
    }
}

#[test]
fn test_map_and_to_vec() {
    let a = bind vec::iter(~[0, 1, 2], _);
    let b = bind map(a, {|i| 2*i}, _);
    let c = to_vec(b);
    assert c == ~[0, 2, 4];
}

#[test]
fn test_map_directly_on_vec() {
    let b = bind map(~[0, 1, 2], {|i| 2*i}, _);
    let c = to_vec(b);
    assert c == ~[0, 2, 4];
}

#[test]
fn test_filter_on_int_range() {
    fn is_even(&&i: int) -> bool {
        return (i % 2) == 0;
    }

    let l = to_vec(bind filter(bind int::range(0, 10, _), is_even, _));
    assert l == ~[0, 2, 4, 6, 8];
}

#[test]
fn test_filter_on_uint_range() {
    fn is_even(&&i: uint) -> bool {
        return (i % 2u) == 0u;
    }

    let l = to_vec(bind filter(bind uint::range(0u, 10u, _), is_even, _));
    assert l == ~[0u, 2u, 4u, 6u, 8u];
}

#[test]
fn test_filter_map() {
    fn negativate_the_evens(&&i: int) -> option<int> {
        if i % 2 == 0 {
            some(-i)
        } else {
            none
        }
    }

    let l = to_vec(bind filter_map(
        bind int::range(0, 5, _), negativate_the_evens, _));
    assert l == ~[0, -2, -4];
}

#[test]
fn test_flat_map_with_option() {
    fn if_even(&&i: int) -> option<int> {
        if (i % 2) == 0 { some(i) }
        else { none }
    }

    let a = bind vec::iter(~[0, 1, 2], _);
    let b = bind flat_map(a, if_even, _);
    let c = to_vec(b);
    assert c == ~[0, 2];
}

#[test]
fn test_flat_map_with_list() {
    fn repeat(&&i: int) -> ~[int] {
        let mut r = ~[];
        int::range(0, i) {|_j| r += ~[i]; }
        r
    }

    let a = bind vec::iter(~[0, 1, 2, 3], _);
    let b = bind flat_map(a, repeat, _);
    let c = to_vec(b);
    debug!{"c = %?", c};
    assert c == ~[1, 2, 2, 3, 3, 3];
}

#[test]
fn test_repeat() {
    let mut c = ~[], i = 0u;
    repeat(5u) {||
        c += ~[(i * i)];
        i += 1u;
    };
    debug!{"c = %?", c};
    assert c == ~[0u, 1u, 4u, 9u, 16u];
}

#[test]
fn test_min() {
    assert min(~[5, 4, 1, 2, 3]) == 1;
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_min_empty() {
    min::<int, ~[int]>(~[]);
}

#[test]
fn test_max() {
    assert max(~[1, 2, 4, 2, 3]) == 4;
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_max_empty() {
    max::<int, ~[int]>(~[]);
}

#[test]
fn test_reversed() {
    assert to_vec(bind reversed(~[1, 2, 3], _)) == ~[3, 2, 1];
}

#[test]
fn test_count() {
    assert count(~[1, 2, 1, 2, 1], 1) == 3u;
}

#[test]
fn test_foldr() {
    fn sub(&&a: int, &&b: int) -> int {
        a - b
    }
    let sum = foldr(~[1, 2, 3, 4], 0, sub);
    assert sum == -2;
}
*/
