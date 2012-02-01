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

fn enumerate<A,IA:iterable<A>>(self: IA, blk: fn(uint, A)) {
    let i = 0u;
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

fn foldl<A,B:copy,IA:iterable<A>>(self: IA, b0: B, blk: fn(B, A) -> B) -> B {
    let b = b0;
    self.iter {|a|
        b = blk(b, a);
    }
    ret b;
}

fn to_list<A:copy,IA:iterable<A>>(self: IA) -> [A] {
    foldl::<A,[A],IA>(self, [], {|r, a| r + [a]})
}

fn repeat(times: uint, blk: fn()) {
    let i = 0u;
    while i < times {
        blk();
        i += 1u;
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
        let r = [];
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
    let c = [],
        i = 0u;
    repeat(5u) {||
        c += [(i * i)];
        i += 1u;
    };
    #debug["c = %?", c];
    assert c == [0u, 1u, 4u, 9u, 16u];
}


