import option::{some, none};

/*
Module: deque

A deque.  Untested as of yet.  Likely buggy.
*/

/*
Iface: t
*/
iface t<T> {
    // Method: size
    fn size() -> uint;
    // Method: add_front
    fn add_front(T);
    // Method: add_back
    fn add_back(T);
    // Method: pop_front
    fn pop_front() -> T;
    // Method: pop_back
    fn pop_back() -> T;
    // Method: peek_front
    fn peek_front() -> T;
    // Method: peek_back
    fn peek_back() -> T;
    // Method: get
    fn get(int) -> T;
}

/*
Section: Functions
*/

/*
Function: create
*/
// FIXME eventually, a proper datatype plus an exported impl would be
// preferrable
fn create<T: copy>() -> t<T> {
    type cell<T> = option::t<T>;

    let initial_capacity: uint = 32u; // 2^5
     /**
      * Grow is only called on full elts, so nelts is also len(elts), unlike
      * elsewhere.
      */
    fn grow<T: copy>(nelts: uint, lo: uint, elts: [mutable cell<T>]) ->
       [mutable cell<T>] {
        assert (nelts == vec::len(elts));
        let rv = [mutable];

        let i = 0u;
        let nalloc = uint::next_power_of_two(nelts + 1u);
        while i < nalloc {
            if i < nelts {
                rv += [mutable elts[(lo + i) % nelts]];
            } else { rv += [mutable none]; }
            i += 1u;
        }

        ret rv;
    }
    fn get<T: copy>(elts: [mutable cell<T>], i: uint) -> T {
        ret alt elts[i] { some(t) { t } _ { fail } };
    }

    type repr<T> = {mutable nelts: uint,
                    mutable lo: uint,
                    mutable hi: uint,
                    mutable elts: [mutable cell<T>]};

    impl <T: copy> of t<T> for repr<T> {
        fn size() -> uint { ret self.nelts; }
        fn add_front(t: T) {
            let oldlo: uint = self.lo;
            if self.lo == 0u {
                self.lo = vec::len(self.elts) - 1u;
            } else { self.lo -= 1u; }
            if self.lo == self.hi {
                self.elts = grow(self.nelts, oldlo, self.elts);
                self.lo = vec::len(self.elts) - 1u;
                self.hi = self.nelts;
            }
            self.elts[self.lo] = some(t);
            self.nelts += 1u;
        }
        fn add_back(t: T) {
            if self.lo == self.hi && self.nelts != 0u {
                self.elts = grow(self.nelts, self.lo, self.elts);
                self.lo = 0u;
                self.hi = self.nelts;
            }
            self.elts[self.hi] = some(t);
            self.hi = (self.hi + 1u) % vec::len(self.elts);
            self.nelts += 1u;
        }
        /**
         * We actually release (turn to none()) the T we're popping so
         * that we don't keep anyone's refcount up unexpectedly.
         */
        fn pop_front() -> T {
            let t: T = get(self.elts, self.lo);
            self.elts[self.lo] = none;
            self.lo = (self.lo + 1u) % vec::len(self.elts);
            self.nelts -= 1u;
            ret t;
        }
        fn pop_back() -> T {
            if self.hi == 0u {
                self.hi = vec::len(self.elts) - 1u;
            } else { self.hi -= 1u; }
            let t: T = get(self.elts, self.hi);
            self.elts[self.hi] = none;
            self.nelts -= 1u;
            ret t;
        }
        fn peek_front() -> T { ret get(self.elts, self.lo); }
        fn peek_back() -> T { ret get(self.elts, self.hi - 1u); }
        fn get(i: int) -> T {
            let idx = (self.lo + (i as uint)) % vec::len(self.elts);
            ret get(self.elts, idx);
        }
    }

    let repr: repr<T> = {
        mutable nelts: 0u,
        mutable lo: 0u,
        mutable hi: 0u,
        mutable elts: vec::init_elt_mut(none, initial_capacity)
    };
    repr as t::<T>
}
