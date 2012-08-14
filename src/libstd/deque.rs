//! A deque. Untested as of yet. Likely buggy

import option::{some, none};
import dvec::{DVec, dvec};

trait t<T> {
    fn size() -> uint;
    fn add_front(T);
    fn add_back(T);
    fn pop_front() -> T;
    fn pop_back() -> T;
    fn peek_front() -> T;
    fn peek_back() -> T;
    fn get(int) -> T;
}

// FIXME (#2343) eventually, a proper datatype plus an exported impl would
// be preferrable.
fn create<T: copy>() -> t<T> {
    type cell<T> = option<T>;

    let initial_capacity: uint = 32u; // 2^5
     /**
      * Grow is only called on full elts, so nelts is also len(elts), unlike
      * elsewhere.
      */
    fn grow<T: copy>(nelts: uint, lo: uint, -elts: ~[mut cell<T>]) ->
       ~[mut cell<T>] {
        assert (nelts == vec::len(elts));
        let mut rv = ~[mut];

        let mut i = 0u;
        let nalloc = uint::next_power_of_two(nelts + 1u);
        while i < nalloc {
            if i < nelts {
                vec::push(rv, elts[(lo + i) % nelts]);
            } else { vec::push(rv, none); }
            i += 1u;
        }

        return rv;
    }
    fn get<T: copy>(elts: DVec<cell<T>>, i: uint) -> T {
        match elts.get_elt(i) { some(t) => t, _ => fail }
    }

    type repr<T> = {mut nelts: uint,
                    mut lo: uint,
                    mut hi: uint,
                    elts: DVec<cell<T>>};

    impl <T: copy> repr<T>: t<T> {
        fn size() -> uint { return self.nelts; }
        fn add_front(t: T) {
            let oldlo: uint = self.lo;
            if self.lo == 0u {
                self.lo = self.elts.len() - 1u;
            } else { self.lo -= 1u; }
            if self.lo == self.hi {
                self.elts.swap(|v| grow(self.nelts, oldlo, v));
                self.lo = self.elts.len() - 1u;
                self.hi = self.nelts;
            }
            self.elts.set_elt(self.lo, some(t));
            self.nelts += 1u;
        }
        fn add_back(t: T) {
            if self.lo == self.hi && self.nelts != 0u {
                self.elts.swap(|v| grow(self.nelts, self.lo, v));
                self.lo = 0u;
                self.hi = self.nelts;
            }
            self.elts.set_elt(self.hi, some(t));
            self.hi = (self.hi + 1u) % self.elts.len();
            self.nelts += 1u;
        }
        /**
         * We actually release (turn to none()) the T we're popping so
         * that we don't keep anyone's refcount up unexpectedly.
         */
        fn pop_front() -> T {
            let t: T = get(self.elts, self.lo);
            self.elts.set_elt(self.lo, none);
            self.lo = (self.lo + 1u) % self.elts.len();
            self.nelts -= 1u;
            return t;
        }
        fn pop_back() -> T {
            if self.hi == 0u {
                self.hi = self.elts.len() - 1u;
            } else { self.hi -= 1u; }
            let t: T = get(self.elts, self.hi);
            self.elts.set_elt(self.hi, none);
            self.nelts -= 1u;
            return t;
        }
        fn peek_front() -> T { return get(self.elts, self.lo); }
        fn peek_back() -> T { return get(self.elts, self.hi - 1u); }
        fn get(i: int) -> T {
            let idx = (self.lo + (i as uint)) % self.elts.len();
            return get(self.elts, idx);
        }
    }

    let repr: repr<T> = {
        mut nelts: 0u,
        mut lo: 0u,
        mut hi: 0u,
        elts:
            dvec::from_vec(
                vec::to_mut(
                    vec::from_elem(initial_capacity, none)))
    };
    repr as t::<T>
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_simple() {
        let d: deque::t<int> = deque::create::<int>();
        assert (d.size() == 0u);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        assert (d.size() == 3u);
        d.add_back(137);
        assert (d.size() == 4u);
        log(debug, d.peek_front());
        assert (d.peek_front() == 42);
        log(debug, d.peek_back());
        assert (d.peek_back() == 137);
        let mut i: int = d.pop_front();
        log(debug, i);
        assert (i == 42);
        i = d.pop_back();
        log(debug, i);
        assert (i == 137);
        i = d.pop_back();
        log(debug, i);
        assert (i == 137);
        i = d.pop_back();
        log(debug, i);
        assert (i == 17);
        assert (d.size() == 0u);
        d.add_back(3);
        assert (d.size() == 1u);
        d.add_front(2);
        assert (d.size() == 2u);
        d.add_back(4);
        assert (d.size() == 3u);
        d.add_front(1);
        assert (d.size() == 4u);
        log(debug, d.get(0));
        log(debug, d.get(1));
        log(debug, d.get(2));
        log(debug, d.get(3));
        assert (d.get(0) == 1);
        assert (d.get(1) == 2);
        assert (d.get(2) == 3);
        assert (d.get(3) == 4);
    }

    fn test_boxes(a: @int, b: @int, c: @int, d: @int) {
        let deq: deque::t<@int> = deque::create::<@int>();
        assert (deq.size() == 0u);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert (deq.size() == 3u);
        deq.add_back(d);
        assert (deq.size() == 4u);
        assert (deq.peek_front() == b);
        assert (deq.peek_back() == d);
        assert (deq.pop_front() == b);
        assert (deq.pop_back() == d);
        assert (deq.pop_back() == c);
        assert (deq.pop_back() == a);
        assert (deq.size() == 0u);
        deq.add_back(c);
        assert (deq.size() == 1u);
        deq.add_front(b);
        assert (deq.size() == 2u);
        deq.add_back(d);
        assert (deq.size() == 3u);
        deq.add_front(a);
        assert (deq.size() == 4u);
        assert (deq.get(0) == a);
        assert (deq.get(1) == b);
        assert (deq.get(2) == c);
        assert (deq.get(3) == d);
    }

    type eqfn<T> = fn@(T, T) -> bool;

    fn test_parameterized<T: copy owned>(
        e: eqfn<T>, a: T, b: T, c: T, d: T) {

        let deq: deque::t<T> = deque::create::<T>();
        assert (deq.size() == 0u);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert (deq.size() == 3u);
        deq.add_back(d);
        assert (deq.size() == 4u);
        assert (e(deq.peek_front(), b));
        assert (e(deq.peek_back(), d));
        assert (e(deq.pop_front(), b));
        assert (e(deq.pop_back(), d));
        assert (e(deq.pop_back(), c));
        assert (e(deq.pop_back(), a));
        assert (deq.size() == 0u);
        deq.add_back(c);
        assert (deq.size() == 1u);
        deq.add_front(b);
        assert (deq.size() == 2u);
        deq.add_back(d);
        assert (deq.size() == 3u);
        deq.add_front(a);
        assert (deq.size() == 4u);
        assert (e(deq.get(0), a));
        assert (e(deq.get(1), b));
        assert (e(deq.get(2), c));
        assert (e(deq.get(3), d));
    }

    enum taggy { one(int), two(int, int), three(int, int, int), }

    enum taggypar<T> {
        onepar(int), twopar(int, int), threepar(int, int, int),
    }

    type reccy = {x: int, y: int, t: taggy};

    #[test]
    fn test() {
        fn inteq(&&a: int, &&b: int) -> bool { return a == b; }
        fn intboxeq(&&a: @int, &&b: @int) -> bool { return a == b; }
        fn taggyeq(a: taggy, b: taggy) -> bool {
            match a {
              one(a1) => match b {
                one(b1) => return a1 == b1,
                _ => return false
              },
              two(a1, a2) => match b {
                two(b1, b2) => return a1 == b1 && a2 == b2,
                _ => return false
              },
              three(a1, a2, a3) => match b {
                three(b1, b2, b3) => return a1 == b1 && a2 == b2 && a3 == b3,
                _ => return false
              }
            }
        }
        fn taggypareq<T>(a: taggypar<T>, b: taggypar<T>) -> bool {
            match a {
              onepar::<T>(a1) => match b {
                onepar::<T>(b1) => return a1 == b1,
                _ => return false
              },
              twopar::<T>(a1, a2) => match b {
                twopar::<T>(b1, b2) => return a1 == b1 && a2 == b2,
                _ => return false
              },
              threepar::<T>(a1, a2, a3) => match b {
                threepar::<T>(b1, b2, b3) => {
                    return a1 == b1 && a2 == b2 && a3 == b3
                }
                _ => return false
              }
            }
        }
        fn reccyeq(a: reccy, b: reccy) -> bool {
            return a.x == b.x && a.y == b.y && taggyeq(a.t, b.t);
        }
        debug!{"*** test boxes"};
        test_boxes(@5, @72, @64, @175);
        debug!{"*** end test boxes"};
        debug!{"test parameterized: int"};
        let eq1: eqfn<int> = inteq;
        test_parameterized::<int>(eq1, 5, 72, 64, 175);
        debug!{"*** test parameterized: @int"};
        let eq2: eqfn<@int> = intboxeq;
        test_parameterized::<@int>(eq2, @5, @72, @64, @175);
        debug!{"*** end test parameterized @int"};
        debug!{"test parameterized: taggy"};
        let eq3: eqfn<taggy> = taggyeq;
        test_parameterized::<taggy>(eq3, one(1), two(1, 2), three(1, 2, 3),
                                    two(17, 42));

        debug!{"*** test parameterized: taggypar<int>"};
        let eq4: eqfn<taggypar<int>> = |x,y| taggypareq::<int>(x, y);
        test_parameterized::<taggypar<int>>(eq4, onepar::<int>(1),
                                            twopar::<int>(1, 2),
                                            threepar::<int>(1, 2, 3),
                                            twopar::<int>(17, 42));
        debug!{"*** end test parameterized: taggypar::<int>"};

        debug!{"*** test parameterized: reccy"};
        let reccy1: reccy = {x: 1, y: 2, t: one(1)};
        let reccy2: reccy = {x: 345, y: 2, t: two(1, 2)};
        let reccy3: reccy = {x: 1, y: 777, t: three(1, 2, 3)};
        let reccy4: reccy = {x: 19, y: 252, t: two(17, 42)};
        let eq5: eqfn<reccy> = reccyeq;
        test_parameterized::<reccy>(eq5, reccy1, reccy2, reccy3, reccy4);
        debug!{"*** end test parameterized: reccy"};
        debug!{"*** done"};
    }
}
