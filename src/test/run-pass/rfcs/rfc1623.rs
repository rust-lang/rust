// run-pass
#![allow(unused_variables)]
#![allow(non_upper_case_globals)]

#![allow(dead_code)]

// very simple test for a 'static static with default lifetime
static STATIC_STR: &str = "&'static str";
const CONST_STR: &str = "&'static str";

// this should be the same as without default:
static EXPLICIT_STATIC_STR: &'static str = "&'static str";
const EXPLICIT_CONST_STR: &'static str = "&'static str";

// a function that elides to an unbound lifetime for both in- and output
fn id_u8_slice(arg: &[u8]) -> &[u8] {
    arg
}

// one with a function, argument elided
static STATIC_SIMPLE_FN: &fn(&[u8]) -> &[u8] = &(id_u8_slice as fn(&[u8]) -> &[u8]);
const CONST_SIMPLE_FN: &fn(&[u8]) -> &[u8] = &(id_u8_slice as fn(&[u8]) -> &[u8]);

// this should be the same as without elision
static STATIC_NON_ELIDED_fN: &for<'a> fn(&'a [u8]) -> &'a [u8] =
    &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);
const CONST_NON_ELIDED_fN: &for<'a> fn(&'a [u8]) -> &'a [u8] =
    &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);

// another function that elides, each to a different unbound lifetime
fn multi_args(a: &u8, b: &u8, c: &u8) {}

static STATIC_MULTI_FN: &fn(&u8, &u8, &u8) = &(multi_args as fn(&u8, &u8, &u8));
const CONST_MULTI_FN: &fn(&u8, &u8, &u8) = &(multi_args as fn(&u8, &u8, &u8));

struct Foo<'a> {
    bools: &'a [bool],
}

static STATIC_FOO: Foo = Foo { bools: &[true, false] };
const CONST_FOO: Foo = Foo { bools: &[true, false] };

type Bar<'a> = Foo<'a>;

static STATIC_BAR: Bar = Bar { bools: &[true, false] };
const CONST_BAR: Bar = Bar { bools: &[true, false] };

type Baz<'a> = fn(&'a [u8]) -> Option<u8>;

fn baz(e: &[u8]) -> Option<u8> {
    e.first().map(|x| *x)
}

static STATIC_BAZ: &Baz = &(baz as Baz);
const CONST_BAZ: &Baz = &(baz as Baz);

static BYTES: &[u8] = &[1, 2, 3];

fn main() {
    // make sure that the lifetime is actually elided (and not defaulted)
    let x = &[1u8, 2, 3];
    STATIC_SIMPLE_FN(x);
    CONST_SIMPLE_FN(x);

    STATIC_BAZ(BYTES); // neees static lifetime
    CONST_BAZ(BYTES);

    // make sure this works with different lifetimes
    let a = &1;
    {
        let b = &2;
        let c = &3;
        CONST_MULTI_FN(a, b, c);
    }
}
