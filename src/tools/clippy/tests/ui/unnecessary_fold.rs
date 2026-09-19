#![warn(clippy::unnecessary_fold)]

fn is_any(acc: bool, x: usize) -> bool {
    acc || x > 2
}

/// Calls which should trigger the `UNNECESSARY_FOLD` lint
fn unnecessary_fold() {
    use std::ops::{Add, Mul};

    // Can be replaced by .any
    let _ = (0..3).fold(false, |acc, x| acc || x > 2);
    //~^ unnecessary_fold

    // Can be replaced by .any (checking suggestion)
    let _ = (0..3).fold(false, |acc, x| is_any(acc, x));
    //~^ redundant_closure

    // Can be replaced by .all
    let _ = (0..3).fold(true, |acc, x| acc && x > 2);
    //~^ unnecessary_fold

    // Can be replaced by .sum
    let _: i32 = (0..3).fold(0, |acc, x| acc + x);
    //~^ unnecessary_fold
    let _: i32 = (0..3).fold(0, Add::add);
    //~^ unnecessary_fold
    let _: i32 = (0..3).fold(0, i32::add);
    //~^ unnecessary_fold

    // Can be replaced by .product
    let _: i32 = (0..3).fold(1, |acc, x| acc * x);
    //~^ unnecessary_fold
    let _: i32 = (0..3).fold(1, Mul::mul);
    //~^ unnecessary_fold
    let _: i32 = (0..3).fold(1, i32::mul);
    //~^ unnecessary_fold
}

/// Should trigger the `UNNECESSARY_FOLD` lint, with an error span including exactly `.fold(...)`
fn unnecessary_fold_span_for_multi_element_chain() {
    let _: bool = (0..3).map(|x| 2 * x).fold(false, |acc, x| acc || x > 2);
    //~^ unnecessary_fold
}

/// Calls which should not trigger the `UNNECESSARY_FOLD` lint
fn unnecessary_fold_should_ignore() {
    let _ = (0..3).fold(true, |acc, x| acc || x > 2);
    let _ = (0..3).fold(false, |acc, x| acc && x > 2);
    let _ = (0..3).fold(1, |acc, x| acc + x);
    let _ = (0..3).fold(0, |acc, x| acc * x);
    let _ = (0..3).fold(0, |acc, x| 1 + acc + x);

    struct Adder;
    impl Adder {
        fn add(lhs: i32, rhs: i32) -> i32 {
            unimplemented!()
        }
        fn mul(lhs: i32, rhs: i32) -> i32 {
            unimplemented!()
        }
    }
    // `add`/`mul` are inherent methods
    let _: i32 = (0..3).fold(0, Adder::add);
    let _: i32 = (0..3).fold(1, Adder::mul);

    trait FakeAdd<Rhs = Self> {
        type Output;
        fn add(self, other: Rhs) -> Self::Output;
    }
    impl FakeAdd for i32 {
        type Output = Self;
        fn add(self, other: i32) -> Self::Output {
            self + other
        }
    }
    trait FakeMul<Rhs = Self> {
        type Output;
        fn mul(self, other: Rhs) -> Self::Output;
    }
    impl FakeMul for i32 {
        type Output = Self;
        fn mul(self, other: i32) -> Self::Output {
            self * other
        }
    }
    // `add`/`mul` come from an unrelated trait
    let _: i32 = (0..3).fold(0, FakeAdd::add);
    let _: i32 = (0..3).fold(1, FakeMul::mul);

    let _ = [(0..2), (0..3)].iter().fold(0, |a, b| a + b.len());
    let _ = [(0..2), (0..3)].iter().fold(1, |a, b| a * b.len());
}

/// Should lint only the line containing the fold
fn unnecessary_fold_over_multiple_lines() {
    let _ = (0..3)
        .map(|x| x + 1)
        .filter(|x| x % 2 == 0)
        .fold(false, |acc, x| acc || x > 2);
    //~^ unnecessary_fold
}

fn issue10000() {
    use std::collections::HashMap;
    use std::hash::BuildHasher;
    use std::ops::{Add, Mul};

    fn anything<T>(_: T) {}
    fn num(_: i32) {}
    fn smoketest_map<S: BuildHasher>(mut map: HashMap<i32, i32, S>) {
        map.insert(0, 0);
        assert_eq!(map.values().fold(0, |x, y| x + y), 0);
        //~^ unnecessary_fold

        // more cases:
        let _ = map.values().fold(0, |x, y| x + y);
        //~^ unnecessary_fold
        let _ = map.values().fold(0, Add::add);
        //~^ unnecessary_fold
        let _ = map.values().fold(1, |x, y| x * y);
        //~^ unnecessary_fold
        let _ = map.values().fold(1, Mul::mul);
        //~^ unnecessary_fold
        let _: i32 = map.values().fold(0, |x, y| x + y);
        //~^ unnecessary_fold
        let _: i32 = map.values().fold(0, Add::add);
        //~^ unnecessary_fold
        let _: i32 = map.values().fold(1, |x, y| x * y);
        //~^ unnecessary_fold
        let _: i32 = map.values().fold(1, Mul::mul);
        //~^ unnecessary_fold
        anything(map.values().fold(0, |x, y| x + y));
        //~^ unnecessary_fold
        anything(map.values().fold(0, Add::add));
        //~^ unnecessary_fold
        anything(map.values().fold(1, |x, y| x * y));
        //~^ unnecessary_fold
        anything(map.values().fold(1, Mul::mul));
        //~^ unnecessary_fold
        num(map.values().fold(0, |x, y| x + y));
        //~^ unnecessary_fold
        num(map.values().fold(0, Add::add));
        //~^ unnecessary_fold
        num(map.values().fold(1, |x, y| x * y));
        //~^ unnecessary_fold
        num(map.values().fold(1, Mul::mul));
        //~^ unnecessary_fold
    }

    smoketest_map(HashMap::new());

    fn add_turbofish_not_necessary() -> i32 {
        (0..3).fold(0, |acc, x| acc + x)
        //~^ unnecessary_fold
    }
    fn mul_turbofish_not_necessary() -> i32 {
        (0..3).fold(1, |acc, x| acc * x)
        //~^ unnecessary_fold
    }
    fn add_turbofish_necessary() -> impl Add {
        (0..3).fold(0, |acc, x| acc + x)
        //~^ unnecessary_fold
    }
    fn mul_turbofish_necessary() -> impl Mul {
        (0..3).fold(1, |acc, x| acc * x)
        //~^ unnecessary_fold
    }
}

fn issue16581() {
    let _ = (2..=3).fold(1, |a, b| a * b);
    //~^ unnecessary_fold
    let _ = (1..=3).fold(0, |a, b| a + b);
    //~^ unnecessary_fold
    let _ = (2..=3).fold(1, |b, a| a * b);
    //~^ unnecessary_fold
    let _ = (1..=3).fold(0, |b, a| a + b);
    //~^ unnecessary_fold

    let _ = (0..3).fold(false, |acc, x| x > 2 || acc);
    //~^ unnecessary_fold
    let _ = (0..3).fold(true, |acc, x| x > 2 && acc);
    //~^ unnecessary_fold
    let _ = (0..3).fold(0, |acc, x| x + acc);
    //~^ unnecessary_fold
    let _ = (0..3).fold(1, |acc, x| x * acc);
    //~^ unnecessary_fold
}

fn wrongly_unmangled_macros() {
    macro_rules! test_expr {
        ($e:expr) => {
            ($e + 1) > 2
        };
    }

    let _ = (0..3).fold(false, |acc: bool, x| acc || test_expr!(x));
    //~^ unnecessary_fold
}

/// Folding over an `Option`'s iterator is `map_or` in disguise (issue #1658)
fn option_fold() {
    let opt: Option<i32> = Some(2);

    // `.iter()`: suggest `opt.as_ref().map_or(...)`
    let _ = opt.iter().fold(10, |acc, x| acc + x);
    //~^ unnecessary_fold

    // `.into_iter()`: `Option` is consumed, suggest plain `map_or`
    let _ = opt.into_iter().fold(10, |acc, x| acc * x);
    //~^ unnecessary_fold

    // `.iter_mut()`: suggest `opt.as_mut().map_or(...)`
    let mut opt_mut: Option<i32> = Some(3);
    let _ = opt_mut.iter_mut().fold(10, |acc, x| acc + *x);
    //~^ unnecessary_fold

    // accumulator unused in the closure body
    let _ = opt.iter().fold(10, |_, x| *x);
    //~^ unnecessary_fold

    // accumulator used more than once: a literal can be duplicated freely
    let _ = opt.iter().fold(2, |acc, x| acc * acc + x);
    //~^ unnecessary_fold

    // a binding of a `Copy` type can also be duplicated freely
    let init = 10;
    let _ = opt.iter().fold(init, |acc, x| acc + x);
    //~^ unnecessary_fold

    // `Option` expression receiver (not a binding)
    let _ = Some(1).into_iter().fold(5, |acc, x| acc - x);
    //~^ unnecessary_fold

    // should NOT lint: `acc` is bound by the enclosing fold's closure, and
    // substituting a closure parameter is not safe when folds are nested
    let _ = (0..3).fold(0, |acc, x| opt.iter().fold(acc, |a, b| a + b) + x);

    // an option fold nested in a standard fold is still linted when its init
    // is a literal
    let _ = (0..3).fold(0, |acc, x| opt.iter().fold(1, |a, b| a + b) + x);
    //~^ unnecessary_fold

    // should NOT lint: a `mut` accumulator is likely reassigned in the body,
    // and substituting into the assignment would not compile
    let _ = opt.iter().fold(0, |mut acc, x| {
        acc += x;
        acc
    });

    // should NOT lint: substituting a call would re-evaluate it
    fn compute() -> i32 {
        42
    }
    let _ = opt.iter().fold(compute(), |acc, x| acc + x);

    // should NOT lint: substituting a non-`Copy` binding would move it twice
    let owned = String::from("a");
    let _ = opt.iter().fold(owned, |acc, x| acc + &x.to_string());

    // should NOT lint: fold over a general iterator with non-literal init
    let _ = (0..3).fold(init, |acc, x| acc + x);

    // should NOT lint: `Result` iterators are out of scope here
    let res: Result<i32, ()> = Ok(1);
    let _ = res.iter().fold(init, |acc, x| acc + x);
}

fn main() {}
