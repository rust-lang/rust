#![feature(plugin)]
#![feature(const_fn)]
#![plugin(clippy)]

#![warn(clippy, clippy_pedantic)]
#![allow(blacklisted_name, unused, print_stdout, non_ascii_literal, new_without_default, new_without_default_derive, missing_docs_in_private_items)]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ops::Mul;
use std::iter::FromIterator;

struct T;

impl T {
    fn add(self, other: T) -> T { self }
    fn drop(&mut self) { }

    fn sub(&self, other: T) -> &T { self } // no error, self is a ref
    fn div(self) -> T { self } // no error, different #arguments
    fn rem(self, other: T) { } // no error, wrong return type

    fn into_u32(self) -> u32 { 0 } // fine
    fn into_u16(&self) -> u16 { 0 }

    fn to_something(self) -> u32 { 0 }

    fn new(self) {}
}

struct Lt<'a> {
    foo: &'a u32,
}

impl<'a> Lt<'a> {
    // The lifetime is different, but that’s irrelevant, see #734
    #[allow(needless_lifetimes)]
    pub fn new<'b>(s: &'b str) -> Lt<'b> { unimplemented!() }
}

struct Lt2<'a> {
    foo: &'a u32,
}

impl<'a> Lt2<'a> {
    // The lifetime is different, but that’s irrelevant, see #734
    pub fn new(s: &str) -> Lt2 { unimplemented!() }
}

struct Lt3<'a> {
    foo: &'a u32,
}

impl<'a> Lt3<'a> {
    // The lifetime is different, but that’s irrelevant, see #734
    pub fn new() -> Lt3<'static> { unimplemented!() }
}

#[derive(Clone,Copy)]
struct U;

impl U {
    fn new() -> Self { U }
    fn to_something(self) -> u32 { 0 } // ok because U is Copy
}

struct V<T> {
    _dummy: T
}

impl<T> V<T> {
    fn new() -> Option<V<T>> { None }
}

impl Mul<T> for T {
    type Output = T;
    fn mul(self, other: T) -> T { self } // no error, obviously
}

/// Utility macro to test linting behavior in `option_methods()`
/// The lints included in `option_methods()` should not lint if the call to map is partially
/// within a macro
macro_rules! opt_map {
    ($opt:expr, $map:expr) => {($opt).map($map)};
}

/// Checks implementation of the following lints:
/// * `OPTION_MAP_UNWRAP_OR`
/// * `OPTION_MAP_UNWRAP_OR_ELSE`
fn option_methods() {
    let opt = Some(1);

    // Check OPTION_MAP_UNWRAP_OR
    // single line case
    let _ = opt.map(|x| x + 1)

               .unwrap_or(0); // should lint even though this call is on a separate line
    // multi line cases
    let _ = opt.map(|x| {
                        x + 1
                    }
              ).unwrap_or(0);
    let _ = opt.map(|x| x + 1)
               .unwrap_or({
                    0
                });
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or(0); // should not lint

    // Check OPTION_MAP_UNWRAP_OR_ELSE
    // single line case
    let _ = opt.map(|x| x + 1)

               .unwrap_or_else(|| 0); // should lint even though this call is on a separate line
    // multi line cases
    let _ = opt.map(|x| {
                        x + 1
                    }
              ).unwrap_or_else(|| 0);
    let _ = opt.map(|x| x + 1)
               .unwrap_or_else(||
                    0
                );
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or_else(|| 0); // should not lint
}

/// Struct to generate false positives for things with .iter()
#[derive(Copy, Clone)]
struct HasIter;

impl HasIter {
    fn iter(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }

    fn iter_mut(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }
}

/// Struct to generate false positive for Iterator-based lints
#[derive(Copy, Clone)]
struct IteratorFalsePositives {
    foo: u32,
}

impl IteratorFalsePositives {
    fn filter(self) -> IteratorFalsePositives {
        self
    }

    fn next(self) -> IteratorFalsePositives {
        self
    }

    fn find(self) -> Option<u32> {
        Some(self.foo)
    }

    fn position(self) -> Option<u32> {
        Some(self.foo)
    }

    fn rposition(self) -> Option<u32> {
        Some(self.foo)
    }

    fn nth(self, n: usize) -> Option<u32> {
        Some(self.foo)
    }

    fn skip(self, _: usize) -> IteratorFalsePositives {
        self
    }
}

#[derive(Copy, Clone)]
struct HasChars;

impl HasChars {
    fn chars(self) -> std::str::Chars<'static> {
        "HasChars".chars()
    }
}

/// Checks implementation of `FILTER_NEXT` lint
fn filter_next() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // check single-line case
    let _ = v.iter().filter(|&x| *x < 0).next();

    // check multi-line case
    let _ = v.iter().filter(|&x| {
                                *x < 0
                            }
                   ).next();

    // check that we don't lint if the caller is not an Iterator
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.filter().next();
}

/// Checks implementation of `SEARCH_IS_SOME` lint
fn search_is_some() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // check `find().is_some()`, single-line
    let _ = v.iter().find(|&x| *x < 0).is_some();

    // check `find().is_some()`, multi-line
    let _ = v.iter().find(|&x| {
                              *x < 0
                          }
                   ).is_some();

    // check `position().is_some()`, single-line
    let _ = v.iter().position(|&x| x < 0).is_some();

    // check `position().is_some()`, multi-line
    let _ = v.iter().position(|&x| {
                                  x < 0
                              }
                   ).is_some();

    // check `rposition().is_some()`, single-line
    let _ = v.iter().rposition(|&x| x < 0).is_some();

    // check `rposition().is_some()`, multi-line
    let _ = v.iter().rposition(|&x| {
                                   x < 0
                               }
                   ).is_some();

    // check that we don't lint if the caller is not an Iterator
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.find().is_some();
    let _ = foo.position().is_some();
    let _ = foo.rposition().is_some();
}

/// Checks implementation of the `OR_FUN_CALL` lint
fn or_fun_call() {
    struct Foo;

    impl Foo {
        fn new() -> Foo { Foo }
    }

    enum Enum {
        A(i32),
    }

    const fn make_const(i: i32) -> i32 { i }

    fn make<T>() -> T { unimplemented!(); }

    let with_enum = Some(Enum::A(1));
    with_enum.unwrap_or(Enum::A(5));

    let with_const_fn = Some(1);
    with_const_fn.unwrap_or(make_const(5));

    let with_constructor = Some(vec![1]);
    with_constructor.unwrap_or(make());

    let with_new = Some(vec![1]);
    with_new.unwrap_or(Vec::new());

    let with_const_args = Some(vec![1]);
    with_const_args.unwrap_or(Vec::with_capacity(12));

    let with_err : Result<_, ()> = Ok(vec![1]);
    with_err.unwrap_or(make());

    let with_err_args : Result<_, ()> = Ok(vec![1]);
    with_err_args.unwrap_or(Vec::with_capacity(12));

    let with_default_trait = Some(1);
    with_default_trait.unwrap_or(Default::default());

    let with_default_type = Some(1);
    with_default_type.unwrap_or(u64::default());

    let with_vec = Some(vec![1]);
    with_vec.unwrap_or(vec![]);

    // FIXME #944: ~|SUGGESTION with_vec.unwrap_or_else(|| vec![]);

    let without_default = Some(Foo);
    without_default.unwrap_or(Foo::new());

    let mut map = HashMap::<u64, String>::new();
    map.entry(42).or_insert(String::new());

    let mut btree = BTreeMap::<u64, String>::new();
    btree.entry(42).or_insert(String::new());

    let stringy = Some(String::from(""));
    let _ = stringy.unwrap_or("".to_owned());
}

/// Checks implementation of `ITER_NTH` lint
fn iter_nth() {
    let mut some_vec = vec![0, 1, 2, 3];
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_vec_deque: VecDeque<_> = some_vec.iter().cloned().collect();

    {
        // Make sure we lint `.iter()` for relevant types
        let bad_vec = some_vec.iter().nth(3);
        let bad_slice = &some_vec[..].iter().nth(3);
        let bad_boxed_slice = boxed_slice.iter().nth(3);
        let bad_vec_deque = some_vec_deque.iter().nth(3);
    }

    {
        // Make sure we lint `.iter_mut()` for relevant types
        let bad_vec = some_vec.iter_mut().nth(3);
    }
    {
        let bad_slice = &some_vec[..].iter_mut().nth(3);
    }
    {
        let bad_vec_deque = some_vec_deque.iter_mut().nth(3);
    }

    // Make sure we don't lint for non-relevant types
    let false_positive = HasIter;
    let ok = false_positive.iter().nth(3);
    let ok_mut = false_positive.iter_mut().nth(3);
}

/// Checks implementation of `ITER_SKIP_NEXT` lint
fn iter_skip_next() {
    let mut some_vec = vec![0, 1, 2, 3];
    let _ = some_vec.iter().skip(42).next();
    let _ = some_vec.iter().cycle().skip(42).next();
    let _ = (1..10).skip(10).next();
    let _ = &some_vec[..].iter().skip(3).next();
    let foo = IteratorFalsePositives { foo : 0 };
    let _ = foo.skip(42).next();
    let _ = foo.filter().skip(42).next();
}

struct GetFalsePositive {
    arr: [u32; 3],
}

impl GetFalsePositive {
    fn get(&self, pos: usize) -> Option<&u32> { self.arr.get(pos) }
    fn get_mut(&mut self, pos: usize) -> Option<&mut u32> { self.arr.get_mut(pos) }
}

/// Checks implementation of `GET_UNWRAP` lint
fn get_unwrap() {
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_slice = &mut [0, 1, 2, 3];
    let mut some_vec = vec![0, 1, 2, 3];
    let mut some_vecdeque: VecDeque<_> = some_vec.iter().cloned().collect();
    let mut some_hashmap: HashMap<u8, char> = HashMap::from_iter(vec![(1, 'a'), (2, 'b')]);
    let mut some_btreemap: BTreeMap<u8, char> = BTreeMap::from_iter(vec![(1, 'a'), (2, 'b')]);
    let mut false_positive = GetFalsePositive { arr: [0, 1, 2] };

    { // Test `get().unwrap()`
        let _ = boxed_slice.get(1).unwrap();
        let _ = some_slice.get(0).unwrap();
        let _ = some_vec.get(0).unwrap();
        let _ = some_vecdeque.get(0).unwrap();
        let _ = some_hashmap.get(&1).unwrap();
        let _ = some_btreemap.get(&1).unwrap();
        let _ = false_positive.get(0).unwrap();
    }

    { // Test `get_mut().unwrap()`
        *boxed_slice.get_mut(0).unwrap() = 1;
        *some_slice.get_mut(0).unwrap() = 1;
        *some_vec.get_mut(0).unwrap() = 1;
        *some_vecdeque.get_mut(0).unwrap() = 1;
        // Check false positives
        *some_hashmap.get_mut(&1).unwrap() = 'b';
        *some_btreemap.get_mut(&1).unwrap() = 'b';
        *false_positive.get_mut(0).unwrap() = 1;
    }
}


#[allow(similar_names)]
fn main() {
    use std::io;

    let opt = Some(0);
    let _ = opt.unwrap();

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();

    res.ok().expect("disaster!");
    // the following should not warn, since `expect` isn't implemented unless
    // the error type implements `Debug`
    let res2: Result<i32, MyError> = Ok(0);
    res2.ok().expect("oh noes!");
    let res3: Result<u32, MyErrorWithParam<u8>>= Ok(0);
    res3.ok().expect("whoof");
    let res4: Result<u32, io::Error> = Ok(0);
    res4.ok().expect("argh");
    let res5: io::Result<u32> = Ok(0);
    res5.ok().expect("oops");
    let res6: Result<u32, &str> = Ok(0);
    res6.ok().expect("meh");
}

struct MyError(()); // doesn't implement Debug

#[derive(Debug)]
struct MyErrorWithParam<T> {
    x: T
}

#[allow(unnecessary_operation)]
fn starts_with() {
    "".chars().next() == Some(' ');
    Some(' ') != "".chars().next();
}

fn str_extend_chars() {
    let abc = "abc";
    let def = String::from("def");
    let mut s = String::new();

    s.push_str(abc);
    s.extend(abc.chars());

    s.push_str("abc");
    s.extend("abc".chars());

    s.push_str(&def);
    s.extend(def.chars());

    s.extend(abc.chars().skip(1));
    s.extend("abc".chars().skip(1));
    s.extend(['a', 'b', 'c'].iter());

    let f = HasChars;
    s.extend(f.chars());
}

fn clone_on_copy() {
    42.clone();

    vec![1].clone(); // ok, not a Copy type
    Some(vec![1]).clone(); // ok, not a Copy type
    (&42).clone();
}

fn clone_on_copy_generic<T: Copy>(t: T) {
    t.clone();

    Some(t).clone();
}

fn clone_on_double_ref() {
    let x = vec![1];
    let y = &&x;
    let z: &Vec<_> = y.clone();

    println!("{:p} {:p}",*y, z);
}

fn single_char_pattern() {
    let x = "foo";
    x.split("x");
    x.split("xx");
    x.split('x');

    let y = "x";
    x.split(y);
    // Not yet testing for multi-byte characters
    // Changing `r.len() == 1` to `r.chars().count() == 1` in `lint_single_char_pattern`
    // should have done this but produced an ICE
    //
    // We may not want to suggest changing these anyway
    // See: https://github.com/rust-lang-nursery/rust-clippy/issues/650#issuecomment-184328984
    x.split("ß");
    x.split("ℝ");
    x.split("💣");
    // Can't use this lint for unicode code points which don't fit in a char
    x.split("❤️");
    x.contains("x");
    x.starts_with("x");
    x.ends_with("x");
    x.find("x");
    x.rfind("x");
    x.rsplit("x");
    x.split_terminator("x");
    x.rsplit_terminator("x");
    x.splitn(0, "x");
    x.rsplitn(0, "x");
    x.matches("x");
    x.rmatches("x");
    x.match_indices("x");
    x.rmatch_indices("x");
    x.trim_left_matches("x");
    x.trim_right_matches("x");

    let h = HashSet::<String>::new();
    h.contains("X"); // should not warn
}

#[allow(result_unwrap_used)]
fn temporary_cstring() {
    use std::ffi::CString;

    CString::new("foo").unwrap().as_ptr();
}

fn iter_clone_collect() {
    let v = [1,2,3,4,5];
    let v2 : Vec<isize> = v.iter().cloned().collect();
    let v3 : HashSet<isize> = v.iter().cloned().collect();
    let v4 : VecDeque<isize> = v.iter().cloned().collect();
}
