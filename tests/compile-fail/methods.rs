#![feature(plugin)]
#![feature(const_fn)]
#![plugin(clippy)]

#![deny(clippy, clippy_pedantic)]
#![allow(blacklisted_name, unused, print_stdout, non_ascii_literal, new_without_default, new_without_default_derive)]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Mul;

struct T;

impl T {
    fn add(self, other: T) -> T { self } //~ERROR defining a method called `add`
    fn drop(&mut self) { } //~ERROR defining a method called `drop`

    fn sub(&self, other: T) -> &T { self } // no error, self is a ref
    fn div(self) -> T { self } // no error, different #arguments
    fn rem(self, other: T) { } // no error, wrong return type

    fn into_u32(self) -> u32 { 0 } // fine
    fn into_u16(&self) -> u16 { 0 } //~ERROR methods called `into_*` usually take self by value

    fn to_something(self) -> u32 { 0 } //~ERROR methods called `to_*` usually take self by reference

    fn new(self) {}
    //~^ ERROR methods called `new` usually take no self
    //~| ERROR methods called `new` usually return `Self`
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
    let _ = opt.map(|x| x + 1) //~  ERROR called `map(f).unwrap_or(a)`
                               //~| NOTE replace `map(|x| x + 1).unwrap_or(0)`
               .unwrap_or(0); // should lint even though this call is on a separate line
    // multi line cases
    let _ = opt.map(|x| { //~ ERROR called `map(f).unwrap_or(a)`
                        x + 1
                    }
              ).unwrap_or(0);
    let _ = opt.map(|x| x + 1) //~ ERROR called `map(f).unwrap_or(a)`
               .unwrap_or({
                    0
                });
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or(0); // should not lint

    // Check OPTION_MAP_UNWRAP_OR_ELSE
    // single line case
    let _ = opt.map(|x| x + 1) //~  ERROR called `map(f).unwrap_or_else(g)`
                               //~| NOTE replace `map(|x| x + 1).unwrap_or_else(|| 0)`
               .unwrap_or_else(|| 0); // should lint even though this call is on a separate line
    // multi line cases
    let _ = opt.map(|x| { //~ ERROR called `map(f).unwrap_or_else(g)`
                        x + 1
                    }
              ).unwrap_or_else(|| 0);
    let _ = opt.map(|x| x + 1) //~ ERROR called `map(f).unwrap_or_else(g)`
               .unwrap_or_else(||
                    0
                );
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or_else(|| 0); // should not lint

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
}

/// Checks implementation of `FILTER_NEXT` lint
fn filter_next() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // check single-line case
    let _ = v.iter().filter(|&x| *x < 0).next();
    //~^ ERROR called `filter(p).next()` on an Iterator.
    //~| NOTE replace `filter(|&x| *x < 0).next()`

    // check multi-line case
    let _ = v.iter().filter(|&x| { //~ERROR called `filter(p).next()` on an Iterator.
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
    //~^ ERROR called `is_some()` after searching
    //~| NOTE replace `find(|&x| *x < 0).is_some()`

    // check `find().is_some()`, multi-line
    let _ = v.iter().find(|&x| { //~ERROR called `is_some()` after searching
                              *x < 0
                          }
                   ).is_some();

    // check `position().is_some()`, single-line
    let _ = v.iter().position(|&x| x < 0).is_some();
    //~^ ERROR called `is_some()` after searching
    //~| NOTE replace `position(|&x| x < 0).is_some()`

    // check `position().is_some()`, multi-line
    let _ = v.iter().position(|&x| { //~ERROR called `is_some()` after searching
                                  x < 0
                              }
                   ).is_some();

    // check `rposition().is_some()`, single-line
    let _ = v.iter().rposition(|&x| x < 0).is_some();
    //~^ ERROR called `is_some()` after searching
    //~| NOTE replace `rposition(|&x| x < 0).is_some()`

    // check `rposition().is_some()`, multi-line
    let _ = v.iter().rposition(|&x| { //~ERROR called `is_some()` after searching
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
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_constructor.unwrap_or_else(make)

    let with_new = Some(vec![1]);
    with_new.unwrap_or(Vec::new());
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_new.unwrap_or_default();

    let with_const_args = Some(vec![1]);
    with_const_args.unwrap_or(Vec::with_capacity(12));
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_const_args.unwrap_or_else(|| Vec::with_capacity(12));

    let with_err : Result<_, ()> = Ok(vec![1]);
    with_err.unwrap_or(make());
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_err.unwrap_or_else(|_| make());

    let with_err_args : Result<_, ()> = Ok(vec![1]);
    with_err_args.unwrap_or(Vec::with_capacity(12));
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_err_args.unwrap_or_else(|_| Vec::with_capacity(12));

    let with_default_trait = Some(1);
    with_default_trait.unwrap_or(Default::default());
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_default_trait.unwrap_or_default();

    let with_default_type = Some(1);
    with_default_type.unwrap_or(u64::default());
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION with_default_type.unwrap_or_default();

    let with_vec = Some(vec![1]);
    with_vec.unwrap_or(vec![]);
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    // FIXME #944: ~|SUGGESTION with_vec.unwrap_or_else(|| vec![]);

    let without_default = Some(Foo);
    without_default.unwrap_or(Foo::new());
    //~^ERROR use of `unwrap_or`
    //~|HELP try this
    //~|SUGGESTION without_default.unwrap_or_else(Foo::new);

    let mut map = HashMap::<u64, String>::new();
    map.entry(42).or_insert(String::new());
    //~^ERROR use of `or_insert` followed by a function call
    //~|HELP try this
    //~|SUGGESTION map.entry(42).or_insert_with(String::new);

    let mut btree = BTreeMap::<u64, String>::new();
    btree.entry(42).or_insert(String::new());
    //~^ERROR use of `or_insert` followed by a function call
    //~|HELP try this
    //~|SUGGESTION btree.entry(42).or_insert_with(String::new);
}

#[allow(similar_names)]
fn main() {
    use std::io;

    let opt = Some(0);
    let _ = opt.unwrap();  //~ERROR used unwrap() on an Option

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();  //~ERROR used unwrap() on a Result

    res.ok().expect("disaster!"); //~ERROR called `ok().expect()`
    // the following should not warn, since `expect` isn't implemented unless
    // the error type implements `Debug`
    let res2: Result<i32, MyError> = Ok(0);
    res2.ok().expect("oh noes!");
    let res3: Result<u32, MyErrorWithParam<u8>>= Ok(0);
    res3.ok().expect("whoof"); //~ERROR called `ok().expect()`
    let res4: Result<u32, io::Error> = Ok(0);
    res4.ok().expect("argh"); //~ERROR called `ok().expect()`
    let res5: io::Result<u32> = Ok(0);
    res5.ok().expect("oops"); //~ERROR called `ok().expect()`
    let res6: Result<u32, &str> = Ok(0);
    res6.ok().expect("meh"); //~ERROR called `ok().expect()`
}

struct MyError(()); // doesn't implement Debug

#[derive(Debug)]
struct MyErrorWithParam<T> {
    x: T
}

#[allow(unnecessary_operation)]
fn starts_with() {
    "".chars().next() == Some(' ');
    //~^ ERROR starts_with
    //~| HELP like this
    //~| SUGGESTION "".starts_with(' ')

    Some(' ') != "".chars().next();
    //~^ ERROR starts_with
    //~| HELP like this
    //~| SUGGESTION !"".starts_with(' ')
}

fn use_extend_from_slice() {
    let mut v : Vec<&'static str> = vec![];
    v.extend(&["Hello", "World"]);
    //~^ ERROR use of `extend`
    //~| HELP try this
    //~| SUGGESTION v.extend_from_slice(&["Hello", "World"]);
    v.extend(&vec!["Some", "more"]);
    //~^ ERROR use of `extend`
    //~| HELP try this
    //~| SUGGESTION v.extend_from_slice(&vec!["Some", "more"]);

    v.extend(vec!["And", "even", "more"].iter());
    //~^ ERROR use of `extend`
    //~| HELP try this
    //FIXME: the suggestion if broken because of the macro
    let o : Option<&'static str> = None;
    v.extend(o);
    v.extend(Some("Bye"));
    v.extend(vec!["Not", "like", "this"]);
    v.extend(["But", "this"].iter());
    //~^ERROR use of `extend
    //~| HELP try this
    //~| SUGGESTION v.extend_from_slice(&["But", "this"]);
}

fn clone_on_copy() {
    42.clone(); //~ERROR using `clone` on a `Copy` type
    vec![1].clone(); // ok, not a Copy type
    Some(vec![1]).clone(); // ok, not a Copy type
}

fn clone_on_copy_generic<T: Copy>(t: T) {
    t.clone(); //~ERROR using `clone` on a `Copy` type
    Some(t).clone(); //~ERROR using `clone` on a `Copy` type
}

fn clone_on_double_ref() {
    let x = vec![1];
    let y = &&x;
    let z: &Vec<_> = y.clone(); //~ERROR using `clone` on a double
                                //~| HELP try dereferencing it
                                //~| SUGGESTION let z: &Vec<_> = (*y).clone();
                                //~^^^ERROR using `clone` on a `Copy` type
    println!("{:p} {:p}",*y, z);
}

fn single_char_pattern() {
    let x = "foo";
    x.split("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.split('x');

    x.split("xx");

    x.split('x');

    let y = "x";
    x.split(y);

    // Not yet testing for multi-byte characters
    // Changing `r.len() == 1` to `r.chars().count() == 1` in `lint_single_char_pattern`
    // should have done this but produced an ICE
    //
    // We may not want to suggest changing these anyway
    // See: https://github.com/Manishearth/rust-clippy/issues/650#issuecomment-184328984
    x.split("ß");
    x.split("ℝ");
    x.split("💣");
    // Can't use this lint for unicode code points which don't fit in a char
    x.split("❤️");

    x.contains("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.contains('x');
    x.starts_with("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.starts_with('x');
    x.ends_with("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.ends_with('x');
    x.find("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.find('x');
    x.rfind("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rfind('x');
    x.rsplit("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rsplit('x');
    x.split_terminator("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.split_terminator('x');
    x.rsplit_terminator("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rsplit_terminator('x');
    x.splitn(0, "x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.splitn(0, 'x');
    x.rsplitn(0, "x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rsplitn(0, 'x');
    x.matches("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.matches('x');
    x.rmatches("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rmatches('x');
    x.match_indices("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.match_indices('x');
    x.rmatch_indices("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.rmatch_indices('x');
    x.trim_left_matches("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.trim_left_matches('x');
    x.trim_right_matches("x");
    //~^ ERROR single-character string constant used as pattern
    //~| HELP try using a char instead:
    //~| SUGGESTION x.trim_right_matches('x');

    let h = HashSet::<String>::new();
    h.contains("X"); // should not warn
}

#[allow(result_unwrap_used)]
fn temporary_cstring() {
    use std::ffi::CString;

    CString::new("foo").unwrap().as_ptr();
    //~^ ERROR you are getting the inner pointer of a temporary `CString`
    //~| NOTE that pointer will be invalid outside this expression
    //~| HELP assign the `CString` to a variable to extend its lifetime
}
