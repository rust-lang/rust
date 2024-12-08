#![allow(dead_code)]

fn non_elidable<'a, 'b>(a: &'a u8, b: &'b u8) -> &'a u8 {
    a
}

// The incorrect case without `for<'a>` is tested for in `rfc1623-3.rs`
static NON_ELIDABLE_FN: &for<'a> fn(&'a u8, &'a u8) -> &'a u8 =
    &(non_elidable as for<'a> fn(&'a u8, &'a u8) -> &'a u8);

struct SomeStruct<'x, 'y, 'z: 'x> {
    foo: &'x Foo<'z>,
    bar: &'x Bar<'z>,
    f: &'y dyn for<'a, 'b> Fn(&'a Foo<'b>) -> &'a Foo<'b>,
}

// Without this, the wf-check will fail early so we'll never see the
// error in SOME_STRUCT's body.
unsafe impl<'x, 'y, 'z: 'x> Sync for SomeStruct<'x, 'y, 'z> {}

fn id<T>(t: T) -> T {
    t
}

static SOME_STRUCT: &SomeStruct = &SomeStruct {
    foo: &Foo { bools: &[false, true] },
    bar: &Bar { bools: &[true, true] },
    f: &id,
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
};

// very simple test for a 'static static with default lifetime
static STATIC_STR: &'static str = "&'static str";
const CONST_STR: &'static str = "&'static str";

// this should be the same as without default:
static EXPLICIT_STATIC_STR: &'static str = "&'static str";
const EXPLICIT_CONST_STR: &'static str = "&'static str";

// a function that elides to an unbound lifetime for both in- and output
fn id_u8_slice(arg: &[u8]) -> &[u8] {
    arg
}

// one with a function, argument elided
static STATIC_SIMPLE_FN: &'static fn(&[u8]) -> &[u8] = &(id_u8_slice as fn(&[u8]) -> &[u8]);
const CONST_SIMPLE_FN: &'static fn(&[u8]) -> &[u8] = &(id_u8_slice as fn(&[u8]) -> &[u8]);

// this should be the same as without elision
static STATIC_NON_ELIDED_fN: &'static for<'a> fn(&'a [u8]) -> &'a [u8] =
    &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);
const CONST_NON_ELIDED_fN: &'static for<'a> fn(&'a [u8]) -> &'a [u8] =
    &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);

// another function that elides, each to a different unbound lifetime
fn multi_args(a: &u8, b: &u8, c: &u8) {}

static STATIC_MULTI_FN: &'static fn(&u8, &u8, &u8) = &(multi_args as fn(&u8, &u8, &u8));
const CONST_MULTI_FN: &'static fn(&u8, &u8, &u8) = &(multi_args as fn(&u8, &u8, &u8));

struct Foo<'a> {
    bools: &'a [bool],
}

static STATIC_FOO: Foo<'static> = Foo { bools: &[true, false] };
const CONST_FOO: Foo<'static> = Foo { bools: &[true, false] };

type Bar<'a> = Foo<'a>;

static STATIC_BAR: Bar<'static> = Bar { bools: &[true, false] };
const CONST_BAR: Bar<'static> = Bar { bools: &[true, false] };

type Baz<'a> = fn(&'a [u8]) -> Option<u8>;

fn baz(e: &[u8]) -> Option<u8> {
    e.first().map(|x| *x)
}

static STATIC_BAZ: &'static Baz<'static> = &(baz as Baz);
const CONST_BAZ: &'static Baz<'static> = &(baz as Baz);

static BYTES: &'static [u8] = &[1, 2, 3];

fn main() {
    let x = &[1u8, 2, 3];
    let y = x;

    // this works, so lifetime < `'static` is valid
    assert_eq!(Some(1), STATIC_BAZ(y));
    assert_eq!(Some(1), CONST_BAZ(y));

    let y = &[1u8, 2, 3];

    STATIC_BAZ(BYTES); // BYTES has static lifetime
    CONST_BAZ(y); // interestingly this does not get reported
}
