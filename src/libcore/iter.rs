iface iterable<A> {
    fn iter(blk: fn(A));
}

impl<A> of iterable<A> for fn@(fn(A)) {
    fn iter(blk: fn(A)) {
        self(blk);
    }
}

// accomodate the fact that int/uint are passed by value by default:
impl of iterable<int> for fn@(fn(int)) {
    fn iter(blk: fn(&&int)) {
        self {|i| blk(i)}
    }
}

impl of iterable<uint> for fn@(fn(uint)) {
    fn iter(blk: fn(&&uint)) {
        self {|i| blk(i)}
    }
}

impl<A> of iterable<A> for [A] {
    fn iter(blk: fn(A)) {
        vec::iter(self, blk)
    }
}

impl<A> of iterable<A> for option<A> {
    fn iter(blk: fn(A)) {
        option::may(self, blk)
    }
}

impl of iterable<char> for str {
    fn iter(blk: fn(&&char)) {
        str::chars_iter(self) { |ch| blk(ch) }
    }
}

fn enumerate<A,IA:iterable<A>>(self: IA, blk: fn(uint, A)) {
    let mut i = 0u;
    self.iter {|a|
        blk(i, a);
        i += 1u;
    }
}

// Here: we have to use fn@ for predicates and map functions, because
// we will be binding them up into a closure.  Disappointing.  A true
// region type system might be able to do better than this.

fn filter<A,IA:iterable<A>>(self: IA, prd: fn@(A) -> bool, blk: fn(A)) {
    self.iter {|a|
        if prd(a) { blk(a) }
    }
}

fn filter_map<A,B,IA:iterable<A>>(self: IA, cnv: fn@(A) -> option<B>,
                                  blk: fn(B)) {
    self.iter {|a|
        alt cnv(a) {
          some(b) { blk(b) }
          none { }
        }
    }
}

fn map<A,B,IA:iterable<A>>(self: IA, cnv: fn@(A) -> B, blk: fn(B)) {
    self.iter {|a|
        let b = cnv(a);
        blk(b);
    }
}

fn flat_map<A,B,IA:iterable<A>,IB:iterable<B>>(
    self: IA, cnv: fn@(A) -> IB, blk: fn(B)) {
    self.iter {|a|
        cnv(a).iter(blk)
    }
}

fn foldl<A,B,IA:iterable<A>>(self: IA, +b0: B, blk: fn(-B, A) -> B) -> B {
    let mut b <- b0;
    self.iter {|a|
        b = blk(b, a);
    }
    ret b;
}

fn foldr<A:copy,B,IA:iterable<A>>(
    self: IA, +b0: B, blk: fn(A, -B) -> B) -> B {

    let mut b <- b0;
    reversed(self) {|a|
        b = blk(a, b);
    }
    ret b;
}

fn to_list<A:copy,IA:iterable<A>>(self: IA) -> [A] {
    foldl::<A,[A],IA>(self, [], {|r, a| r + [a]})
}

// FIXME: This could be made more efficient with an riterable interface
// #2005
fn reversed<A:copy,IA:iterable<A>>(self: IA, blk: fn(A)) {
    vec::riter(to_list(self), blk)
}

fn count<A,IA:iterable<A>>(self: IA, x: A) -> uint {
    foldl(self, 0u) {|count, value|
        if value == x {
            count + 1u
        } else {
            count
        }
    }
}

fn repeat(times: uint, blk: fn()) {
    let mut i = 0u;
    while i < times {
        blk();
        i += 1u;
    }
}

fn min<A:copy,IA:iterable<A>>(self: IA) -> A {
    alt foldl::<A,option<A>,IA>(self, none) {|a, b|
        alt a {
          some(a_) if a_ < b {
            // FIXME: Not sure if this is successfully optimized to a move
            // #2005
            a
          }
          _ { some(b) }
        }
    } {
        some(val) { val }
        none { fail "min called on empty iterator" }
    }
}

fn max<A:copy,IA:iterable<A>>(self: IA) -> A {
    alt foldl::<A,option<A>,IA>(self, none) {|a, b|
        alt a {
          some(a_) if a_ > b {
            // FIXME: Not sure if this is successfully optimized to a move
            // #2005
            a
          }
          _ { some(b) }
        }
    } {
        some(val) { val }
        none { fail "max called on empty iterator" }
    }
}

#[test]
fn test_enumerate() {
    enumerate(["0", "1", "2"]) {|i,j|
        assert #fmt["%u",i] == j;
    }
}

#[test]
fn test_map_and_to_list() {
    let a = bind vec::iter([0, 1, 2], _);
    let b = bind map(a, {|i| i*2}, _);
    let c = to_list(b);
    assert c == [0, 2, 4];
}

#[test]
fn test_map_directly_on_vec() {
    let b = bind map([0, 1, 2], {|i| i*2}, _);
    let c = to_list(b);
    assert c == [0, 2, 4];
}

#[test]
fn test_filter_on_int_range() {
    fn is_even(&&i: int) -> bool {
        ret (i % 2) == 0;
    }

    let l = to_list(bind filter(bind int::range(0, 10, _), is_even, _));
    assert l == [0, 2, 4, 6, 8];
}

#[test]
fn test_filter_on_uint_range() {
    fn is_even(&&i: uint) -> bool {
        ret (i % 2u) == 0u;
    }

    let l = to_list(bind filter(bind uint::range(0u, 10u, _), is_even, _));
    assert l == [0u, 2u, 4u, 6u, 8u];
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

    let l = to_list(bind filter_map(
        bind int::range(0, 5, _), negativate_the_evens, _));
    assert l == [0, -2, -4];
}

#[test]
fn test_flat_map_with_option() {
    fn if_even(&&i: int) -> option<int> {
        if (i % 2) == 0 { some(i) }
        else { none }
    }

    let a = bind vec::iter([0, 1, 2], _);
    let b = bind flat_map(a, if_even, _);
    let c = to_list(b);
    assert c == [0, 2];
}

#[test]
fn test_flat_map_with_list() {
    fn repeat(&&i: int) -> [int] {
        let mut r = [];
        int::range(0, i) {|_j| r += [i]; }
        r
    }

    let a = bind vec::iter([0, 1, 2, 3], _);
    let b = bind flat_map(a, repeat, _);
    let c = to_list(b);
    #debug["c = %?", c];
    assert c == [1, 2, 2, 3, 3, 3];
}

#[test]
fn test_repeat() {
    let mut c = [], i = 0u;
    repeat(5u) {||
        c += [(i * i)];
        i += 1u;
    };
    #debug["c = %?", c];
    assert c == [0u, 1u, 4u, 9u, 16u];
}

#[test]
fn test_min() {
    assert min([5, 4, 1, 2, 3]) == 1;
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_min_empty() {
    min::<int, [int]>([]);
}

#[test]
fn test_max() {
    assert max([1, 2, 4, 2, 3]) == 4;
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_max_empty() {
    max::<int, [int]>([]);
}

#[test]
fn test_reversed() {
    assert to_list(bind reversed([1, 2, 3], _)) == [3, 2, 1];
}

#[test]
fn test_count() {
    assert count([1, 2, 1, 2, 1], 1) == 3u;
}

#[test]
fn test_foldr() {
    fn sub(&&a: int, -b: int) -> int {
        a - b
    }
    let sum = foldr([1, 2, 3, 4], 0, sub);
    assert sum == -2;
}