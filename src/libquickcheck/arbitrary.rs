// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num;
use std::str;
use rand::Rng;

/// `Gen` wraps a `rand::Rng` with parameters to control the distribution of
/// random values.
///
/// A value with type satisfying the `Gen` trait can be constructed with the
/// `gen` function in this crate.
pub trait Gen : Rng {
    fn size(&self) -> uint;
}

/// StdGen is the default implementation of `Gen`.
///
/// Values of type `StdGen` can be created with the `gen` function in this
/// crate.
pub struct StdGen<R> {
    rng: R,
    size: uint,
}

impl<R: Rng> StdGen<R> {
    /// Returns a `Gen` with the given configuration using any random number
    /// generator.
    ///
    /// The `size` parameter controls the size of random values generated.
    /// For example, it specifies the maximum length of a randomly generator vector
    /// and also will specify the maximum magnitude of a randomly generated number.
    pub fn new(rng: R, size: uint) -> StdGen<R> {
        StdGen { rng: rng, size: size }
    }
}

impl<R: Rng> Rng for StdGen<R> {
    fn next_u32(&mut self) -> u32 { self.rng.next_u32() }

    // some RNGs implement these more efficiently than the default, so
    // we might as well defer to them.
    fn next_u64(&mut self) -> u64 { self.rng.next_u64() }
    fn fill_bytes(&mut self, dest: &mut [u8]) { self.rng.fill_bytes(dest) }
}

impl<R: Rng> Gen for StdGen<R> {
    fn size(&self) -> uint { self.size }
}

/// `~Shrinker` is an existential type that represents an arbitrary iterator
/// by satisfying the `Iterator` trait.
///
/// This makes writing shrinkers easier.
/// You should not have to implement this trait directly. By default, all
/// types which implement the `Iterator` trait also implement the `Shrinker`
/// trait.
///
/// The `A` type variable corresponds to the elements yielded by the iterator.
pub trait Shrinker<A> {
    /// Wraps `<A: Iterator>.next()`.
    fn next_shrink(&mut self) -> Option<A>;
}

impl<A> Iterator<A> for Box<Shrinker<A>> {
    fn next(&mut self) -> Option<A> { self.next_shrink() }
}

impl<T, A: Iterator<T>> Shrinker<T> for A {
    fn next_shrink(&mut self) -> Option<T> { self.next() }
}

/// A shrinker than yields no values.
pub struct EmptyShrinker<A>;

impl<A> EmptyShrinker<A> {
    /// Creates a shrinker with zero elements.
    pub fn new() -> Box<Shrinker<A>> {
        box EmptyShrinker::<A> as Box<Shrinker<A>>
    }
}

impl<A> Iterator<A> for EmptyShrinker<A> {
    fn next(&mut self) -> Option<A> { None }
}

/// A shrinker than yields a single value.
pub struct SingleShrinker<A> {
    value: Option<A>
}

impl<A> SingleShrinker<A> {
    /// Creates a shrinker with a single element.
    pub fn new(value: A) -> Box<Shrinker<A>> {
        box SingleShrinker { value: Some(value) } as Box<Shrinker<A>>
    }
}

impl<A> Iterator<A> for SingleShrinker<A> {
    fn next(&mut self) -> Option<A> { self.value.take() }
}

/// `Arbitrary` describes types whose values can be randomly generated and
/// shrunk.
///
/// Aside from shrinking, `Arbitrary` is different from the `std::Rand` trait
/// in that it uses a `Gen` to control the distribution of random values.
///
/// As of now, all types that implement `Arbitrary` must also implement
/// `Clone`. (I'm not sure if this is a permanent restriction.)
///
/// They must also be sendable since every test is run inside its own task.
/// (This permits failures to include task failures.)
pub trait Arbitrary : Clone + Send {
    fn arbitrary<G: Gen>(g: &mut G) -> Self;
    fn shrink(&self) -> Box<Shrinker<Self>> {
        EmptyShrinker::new()
    }
}

impl Arbitrary for () {
    fn arbitrary<G: Gen>(_: &mut G) -> () { () }
}

impl Arbitrary for bool {
    fn arbitrary<G: Gen>(g: &mut G) -> bool { g.gen() }
    fn shrink(&self) -> Box<Shrinker<bool>> {
        match *self {
            true => SingleShrinker::new(false),
            false => EmptyShrinker::new(),
        }
    }
}

impl<A: Arbitrary> Arbitrary for Option<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Option<A> {
        if g.gen() {
            None
        } else {
            Some(Arbitrary::arbitrary(g))
        }
    }

    fn shrink(&self)  -> Box<Shrinker<Option<A>>> {
        match *self {
            None => {
                EmptyShrinker::new()
            }
            Some(ref x) => {
                let chain = SingleShrinker::new(None).chain(x.shrink().map(Some));
                box chain as Box<Shrinker<Option<A>>>
            }
        }
    }
}

impl<A: Arbitrary, B: Arbitrary> Arbitrary for Result<A, B> {
    fn arbitrary<G: Gen>(g: &mut G) -> Result<A, B> {
        if g.gen() {
            Ok(Arbitrary::arbitrary(g))
        } else {
            Err(Arbitrary::arbitrary(g))
        }
    }

    fn shrink(&self) -> Box<Shrinker<Result<A, B>>> {
        match *self {
            Ok(ref x) => {
                let xs: Box<Shrinker<A>> = x.shrink();
                let tagged = xs.map::<'static, Result<A, B>>(Ok);
                box tagged as Box<Shrinker<Result<A, B>>>
            }
            Err(ref x) => {
                let xs: Box<Shrinker<B>> = x.shrink();
                let tagged = xs.map::<'static, Result<A, B>>(Err);
                box tagged as Box<Shrinker<Result<A, B>>>
            }
        }
    }
}

impl<A: Arbitrary, B: Arbitrary> Arbitrary for (A, B) {
    fn arbitrary<G: Gen>(g: &mut G) -> (A, B) {
        return (Arbitrary::arbitrary(g), Arbitrary::arbitrary(g))
    }

    // Shrinking a tuple is done by shrinking the first element and generating
    // a new tuple with each shrunk element from the first along with a copy of
    // the given second element. Vice versa for the second element. More
    // precisely:
    //
    //     shrink((a, b)) =
    //         let (sa, sb) = (a.shrink(), b.shrink());
    //         vec!((sa1, b), ..., (saN, b), (a, sb1), ..., (a, sbN))
    //
    fn shrink(&self) -> Box<Shrinker<(A, B)>> {
        let (ref a, ref b) = *self;
        let sas = a.shrink().scan(b, |b, a| {
            Some((a, b.clone()))
        });
        let sbs = b.shrink().scan(a, |a, b| {
            Some((a.clone(), b))
        });
        box sas.chain(sbs) as Box<Shrinker<(A, B)>>
    }
}

impl<A: Arbitrary, B: Arbitrary, C: Arbitrary> Arbitrary for (A, B, C) {
    fn arbitrary<G: Gen>(g: &mut G) -> (A, B, C) {
        return (
            Arbitrary::arbitrary(g),
            Arbitrary::arbitrary(g),
            Arbitrary::arbitrary(g),
        )
    }

    fn shrink(&self) -> Box<Shrinker<(A, B, C)>> {
        let (ref a, ref b, ref c) = *self;
        let sas = a.shrink().scan((b, c), |&(b, c), a| {
            Some((a, b.clone(), c.clone()))
        });
        let sbs = b.shrink().scan((a, c), |&(a, c), b| {
            Some((a.clone(), b, c.clone()))
        });
        let scs = c.shrink().scan((a, b), |&(a, b), c| {
            Some((a.clone(), b.clone(), c))
        });
        box sas.chain(sbs).chain(scs) as Box<Shrinker<(A, B, C)>>
    }
}

impl<A: Arbitrary> Arbitrary for Vec<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Vec<A> {
        let size = { let s = g.size(); g.gen_range(0, s) };
        Vec::from_fn(size, |_| Arbitrary::arbitrary(g))
    }

    fn shrink(&self) -> Box<Shrinker<Vec<A>>> {
        if self.len() == 0 {
            return EmptyShrinker::new()
        }

        // Start the shrunk values with an empty vector.
        let mut xs: Vec<Vec<A>> = vec!(vec!());

        // Explore the space of different sized vectors without shrinking
        // any of the elements.
        let mut k = self.len() / 2;
        while k > 0 {
            xs.push_all_move(shuffle_vec(self.as_slice(), k));
            k = k / 2;
        }

        // Now explore the space of vectors where each element is shrunk
        // in turn. A new vector is generated for each shrunk value of each
        // element.
        for (i, x) in self.iter().enumerate() {
            for sx in x.shrink() {
                let mut change_one = self.clone();
                *change_one.get_mut(i) = sx;
                xs.push(change_one);
            }
        }
        box xs.move_iter() as Box<Shrinker<Vec<A>>>
    }
}

impl Arbitrary for StrBuf {
    fn arbitrary<G: Gen>(g: &mut G) -> StrBuf {
        let size = { let s = g.size(); g.gen_range(0, s) };
        g.gen_ascii_str(size).to_strbuf()
    }

    fn shrink(&self) -> Box<Shrinker<StrBuf>> {
        // Shrink a string by shrinking a vector of its characters.
        let chars: Vec<char> = self.as_slice().chars().collect();
        box chars.shrink().map(|x| x.move_iter().collect::<StrBuf>()) as Box<Shrinker<StrBuf>>
    }
}

impl Arbitrary for ~str {
    fn arbitrary<G: Gen>(g: &mut G) -> ~str {
        let size = { let s = g.size(); g.gen_range(0, s) };
        g.gen_ascii_str(size)
    }

    fn shrink(&self) -> Box<Shrinker<~str>> {
        // Shrink a string by shrinking a vector of its characters.
        let chars: Vec<char> = self.chars().collect();
        let mut strs: Vec<~str> = vec!();
        for x in chars.shrink() {
            strs.push(str::from_chars(x.as_slice()));
        }
        box strs.move_iter() as Box<Shrinker<~str>>
    }
}

impl Arbitrary for char {
    fn arbitrary<G: Gen>(g: &mut G) -> char { g.gen() }

    fn shrink(&self) -> Box<Shrinker<char>> {
        // No char shrinking for now.
        EmptyShrinker::new()
    }
}

impl Arbitrary for int {
    fn arbitrary<G: Gen>(g: &mut G) -> int {
        let s = g.size(); g.gen_range(-(s as int), s as int)
    }
    fn shrink(&self) -> Box<Shrinker<int>> {
        SignedShrinker::new(*self)
    }
}

impl Arbitrary for i8 {
    fn arbitrary<G: Gen>(g: &mut G) -> i8 {
        let s = g.size(); g.gen_range(-(s as i8), s as i8)
    }
    fn shrink(&self) -> Box<Shrinker<i8>> {
        SignedShrinker::new(*self)
    }
}

impl Arbitrary for i16 {
    fn arbitrary<G: Gen>(g: &mut G) -> i16 {
        let s = g.size(); g.gen_range(-(s as i16), s as i16)
    }
    fn shrink(&self) -> Box<Shrinker<i16>> {
        SignedShrinker::new(*self)
    }
}

impl Arbitrary for i32 {
    fn arbitrary<G: Gen>(g: &mut G) -> i32 {
        let s = g.size(); g.gen_range(-(s as i32), s as i32)
    }
    fn shrink(&self) -> Box<Shrinker<i32>> {
        SignedShrinker::new(*self)
    }
}

impl Arbitrary for i64 {
    fn arbitrary<G: Gen>(g: &mut G) -> i64 {
        let s = g.size(); g.gen_range(-(s as i64), s as i64)
    }
    fn shrink(&self) -> Box<Shrinker<i64>> {
        SignedShrinker::new(*self)
    }
}

impl Arbitrary for uint {
    fn arbitrary<G: Gen>(g: &mut G) -> uint {
        let s = g.size(); g.gen_range(0, s)
    }
    fn shrink(&self) -> Box<Shrinker<uint>> {
        UnsignedShrinker::new(*self)
    }
}

impl Arbitrary for u8 {
    fn arbitrary<G: Gen>(g: &mut G) -> u8 {
        let s = g.size(); g.gen_range(0, s as u8)
    }
    fn shrink(&self) -> Box<Shrinker<u8>> {
        UnsignedShrinker::new(*self)
    }
}

impl Arbitrary for u16 {
    fn arbitrary<G: Gen>(g: &mut G) -> u16 {
        let s = g.size(); g.gen_range(0, s as u16)
    }
    fn shrink(&self) -> Box<Shrinker<u16>> {
        UnsignedShrinker::new(*self)
    }
}

impl Arbitrary for u32 {
    fn arbitrary<G: Gen>(g: &mut G) -> u32 {
        let s = g.size(); g.gen_range(0, s as u32)
    }
    fn shrink(&self) -> Box<Shrinker<u32>> {
        UnsignedShrinker::new(*self)
    }
}

impl Arbitrary for u64 {
    fn arbitrary<G: Gen>(g: &mut G) -> u64 {
        let s = g.size(); g.gen_range(0, s as u64)
    }
    fn shrink(&self) -> Box<Shrinker<u64>> {
        UnsignedShrinker::new(*self)
    }
}

impl Arbitrary for f32 {
    fn arbitrary<G: Gen>(g: &mut G) -> f32 {
        let s = g.size(); g.gen_range(-(s as f32), s as f32)
    }
    fn shrink(&self) -> Box<Shrinker<f32>> {
        let it = SignedShrinker::new(self.to_i32().unwrap());
        box it.map(|x| x.to_f32().unwrap()) as Box<Shrinker<f32>>
    }
}

impl Arbitrary for f64 {
    fn arbitrary<G: Gen>(g: &mut G) -> f64 {
        let s = g.size(); g.gen_range(-(s as f64), s as f64)
    }
    fn shrink(&self) -> Box<Shrinker<f64>> {
        let it = SignedShrinker::new(self.to_i64().unwrap());
        box it.map(|x| x.to_f64().unwrap()) as Box<Shrinker<f64>>
    }
}

/// Returns a sequence of vectors with each contiguous run of elements of
/// length `k` removed.
fn shuffle_vec<A: Clone>(xs: &[A], k: uint) -> Vec<Vec<A>> {
    fn shuffle<A: Clone>(xs: &[A], k: uint, n: uint) -> Vec<Vec<A>> {
        if k > n {
            return vec!()
        }
        let xs1: Vec<A> = xs.slice_to(k).iter().map(|x| x.clone()).collect();
        let xs2: Vec<A> = xs.slice_from(k).iter().map(|x| x.clone()).collect();
        if xs2.len() == 0 {
            return vec!(vec!())
        }

        let cat = |x: &Vec<A>| {
            let mut pre = xs1.clone();
            pre.push_all_move(x.clone());
            pre
        };
        let shuffled = shuffle(xs2.as_slice(), k, n-k);
        let mut more: Vec<Vec<A>> = shuffled.iter().map(cat).collect();
        more.unshift(xs2);
        more
    }
    shuffle(xs, k, xs.len())
}

fn half<A: Primitive>(x: A) -> A { x / num::cast(2).unwrap() }

struct SignedShrinker<A> {
    x: A,
    i: A,
}

impl<A: Primitive + Signed> SignedShrinker<A> {
    fn new(x: A) -> Box<Shrinker<A>> {
        if x.is_zero() {
            EmptyShrinker::<A>::new()
        } else {
            let shrinker = SignedShrinker {
                x: x,
                i: half(x),
            };
            if shrinker.i.is_negative() {
                let start = vec![num::zero(), shrinker.x.abs()].move_iter();
                box start.chain(shrinker) as Box<Shrinker<A>>
            } else {
                box { vec![num::zero()] }.move_iter().chain(shrinker) as Box<Shrinker<A>>
            }
        }
    }
}

impl<A: Primitive + Signed> Iterator<A> for SignedShrinker<A> {
    fn next(&mut self) -> Option<A> {
        if (self.x - self.i).abs() < self.x.abs() {
            let result = Some(self.x - self.i);
            self.i = half(self.i);
            result
        } else {
            None
        }
    }
}

struct UnsignedShrinker<A> {
    x: A,
    i: A,
}

impl<A: Primitive + Unsigned> UnsignedShrinker<A> {
    fn new(x: A) -> Box<Shrinker<A>> {
        if x.is_zero() {
            EmptyShrinker::<A>::new()
        } else {
            box { vec![num::zero()] }.move_iter().chain(
                UnsignedShrinker {
                    x: x,
                    i: half(x),
                }
            ) as Box<Shrinker<A>>
        }
    }
}

impl<A: Primitive + Unsigned> Iterator<A> for UnsignedShrinker<A> {
    fn next(&mut self) -> Option<A> {
        if self.x - self.i < self.x {
            let result = Some(self.x - self.i);
            self.i = half(self.i);
            result
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Show;
    use std::hash::Hash;
    use std::iter;
    use collections::HashSet;
    use rand;
    use super::Arbitrary;

    // Arbitrary testing. (Not much here. What else can I reasonably test?)
    #[test]
    fn arby_unit() {
        assert_eq!(arby::<()>(), ());
    }

    #[test]
    fn arby_int() {
        rep(|| { let n: int = arby(); assert!(n >= -5 && n <= 5); } );
    }

    #[test]
    fn arby_uint() {
        rep(|| { let n: uint = arby(); assert!(n <= 5); } );
    }

    fn arby<A: super::Arbitrary>() -> A {
        super::Arbitrary::arbitrary(&mut gen())
    }

    fn gen() -> super::StdGen<rand::TaskRng> {
        super::StdGen::new(rand::task_rng(), 5)
    }

    fn rep(f: ||) {
        for _ in iter::range(0, 100) {
            f()
        }
    }

    // Shrink testing.
    #[test]
    fn unit() {
        eq((), vec!());
    }

    #[test]
    fn bools() {
        eq(false, vec!());
        eq(true, vec!(false));
    }

    #[test]
    fn options() {
        eq(None::<()>, vec!());
        eq(Some(false), vec!(None));
        eq(Some(true), vec!(None, Some(false)));
    }

    #[test]
    fn results() {
        // FIXME #14097: Result<A, B> doesn't implement the Hash
        // trait, so these tests depends on the order of shrunk
        // results. Ug.
        ordered_eq(Ok::<bool, ()>(true), vec!(Ok(false)));
        ordered_eq(Err::<(), bool>(true), vec!(Err(false)));
    }

    #[test]
    fn tuples() {
        eq((false, false), vec!());
        eq((true, false), vec!((false, false)));
        eq((true, true), vec!((false, true), (true, false)));
    }

    #[test]
    fn triples() {
        eq((false, false, false), vec!());
        eq((true, false, false), vec!((false, false, false)));
        eq((true, true, false),
           vec!((false, true, false), (true, false, false)));
    }

    #[test]
    fn ints() {
        // FIXME #14097: Test overflow?
        eq(5i, vec!(0, 3, 4));
        eq(-5i, vec!(5, 0, -3, -4));
        eq(0i, vec!());
    }

    #[test]
    fn ints8() {
        eq(5i8, vec!(0, 3, 4));
        eq(-5i8, vec!(5, 0, -3, -4));
        eq(0i8, vec!());
    }

    #[test]
    fn ints16() {
        eq(5i16, vec!(0, 3, 4));
        eq(-5i16, vec!(5, 0, -3, -4));
        eq(0i16, vec!());
    }

    #[test]
    fn ints32() {
        eq(5i32, vec!(0, 3, 4));
        eq(-5i32, vec!(5, 0, -3, -4));
        eq(0i32, vec!());
    }

    #[test]
    fn ints64() {
        eq(5i64, vec!(0, 3, 4));
        eq(-5i64, vec!(5, 0, -3, -4));
        eq(0i64, vec!());
    }

    #[test]
    fn uints() {
        eq(5u, vec!(0, 3, 4));
        eq(0u, vec!());
    }

    #[test]
    fn uints8() {
        eq(5u8, vec!(0, 3, 4));
        eq(0u8, vec!());
    }

    #[test]
    fn uints16() {
        eq(5u16, vec!(0, 3, 4));
        eq(0u16, vec!());
    }

    #[test]
    fn uints32() {
        eq(5u32, vec!(0, 3, 4));
        eq(0u32, vec!());
    }

    #[test]
    fn uints64() {
        eq(5u64, vec!(0, 3, 4));
        eq(0u64, vec!());
    }

    #[test]
    fn vecs() {
        eq({let it: Vec<int> = vec!(); it}, vec!());
        eq({let it: Vec<Vec<int>> = vec!(vec!()); it}, vec!(vec!()));
        eq(vec!(1), vec!(vec!(), vec!(0)));
        eq(vec!(11), vec!(vec!(), vec!(0), vec!(6), vec!(9), vec!(10)));
        eq(
            vec!(3, 5),
            vec!(vec!(), vec!(5), vec!(3), vec!(0,5), vec!(2,5),
                 vec!(3,0), vec!(3,3), vec!(3,4))
        );
    }

    #[test]
    fn chars() {
        eq('a', vec!());
    }

    #[test]
    fn strs() {
        eq("".to_owned(), vec!());
        eq("A".to_owned(), vec!("".to_owned()));
        eq("ABC".to_owned(), vec!("".to_owned(),
                                 "AB".to_owned(),
                                 "BC".to_owned(),
                                 "AC".to_owned()));
    }

    // All this jazz is for testing set equality on the results of a shrinker.
    fn eq<A: Arbitrary + TotalEq + Show + Hash>(s: A, v: Vec<A>) {
        let (left, right) = (shrunk(s), set(v));
        assert!(left.eq(&right) && right.eq(&left));
    }
    fn shrunk<A: Arbitrary + TotalEq + Hash>(s: A) -> HashSet<A> {
        set(s.shrink().collect())
    }
    fn set<A: TotalEq + Hash>(xs: Vec<A>) -> HashSet<A> {
        xs.move_iter().collect()
    }

    fn ordered_eq<A: Arbitrary + TotalEq + Show>(s: A, v: Vec<A>) {
        let (left, right) = (s.shrink().collect::<Vec<A>>(), v);
        assert!(left.eq(&right) && right.eq(&left));
    }
}
