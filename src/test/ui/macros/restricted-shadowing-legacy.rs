// Legend:
// `N` - number of combination, from 0 to 4*4*4=64
// `Outer < Invoc` means that expansion that produced macro definition `Outer`
// is a strict ancestor of expansion that produced macro definition `Inner`.
// `>`, `=` and `Unordered` mean "strict descendant", "same" and
// "not in ordering relation" for parent expansions.
// `+` - possible configuration
// `-` - configuration impossible due to properties of partial ordering
// `-?` - configuration impossible due to block/scope syntax
// `+?` - configuration possible only with legacy scoping

//  N | Outer ~ Invoc | Invoc ~ Inner | Outer ~ Inner | Possible |
//  1 |       <       |       <       |       <       |    +     |
//  2 |       <       |       <       |       =       |    -     |
//  3 |       <       |       <       |       >       |    -     |
//  4 |       <       |       <       |   Unordered   |    -     |
//  5 |       <       |       =       |       <       |    +     |
//  6 |       <       |       =       |       =       |    -     |
//  7 |       <       |       =       |       >       |    -     |
//  8 |       <       |       =       |   Unordered   |    -     |
//  9 |       <       |       >       |       <       |    +     |
// 10 |       <       |       >       |       =       |    +     |
// 11 |       <       |       >       |       >       |    -?    |
// 12 |       <       |       >       |   Unordered   |    -?    |
// 13 |       <       |   Unordered   |       <       |    +     |
// 14 |       <       |   Unordered   |       =       |    -     |
// 15 |       <       |   Unordered   |       >       |    -     |
// 16 |       <       |   Unordered   |   Unordered   |    -?    |
// 17 |       =       |       <       |       <       |    +     |
// 18 |       =       |       <       |       =       |    -     |
// 19 |       =       |       <       |       >       |    -     |
// 20 |       =       |       <       |   Unordered   |    -     |
// 21 |       =       |       =       |       <       |    -     |
// 22 |       =       |       =       |       =       |    +     |
// 23 |       =       |       =       |       >       |    -     |
// 24 |       =       |       =       |   Unordered   |    -     |
// 25 |       =       |       >       |       <       |    -     |
// 26 |       =       |       >       |       =       |    -     |
// 27 |       =       |       >       |       >       |    -?    |
// 28 |       =       |       >       |   Unordered   |    -     |
// 29 |       =       |   Unordered   |       <       |    -     |
// 30 |       =       |   Unordered   |       =       |    -     |
// 31 |       =       |   Unordered   |       >       |    -     |
// 32 |       =       |   Unordered   |   Unordered   |    -?    |
// 33 |       >       |       <       |       <       |    +?    |
// 34 |       >       |       <       |       =       |    +?    |
// 35 |       >       |       <       |       >       |    +?    |
// 36 |       >       |       <       |   Unordered   |    +     |
// 37 |       >       |       =       |       <       |    -     |
// 38 |       >       |       =       |       =       |    -     |
// 39 |       >       |       =       |       >       |    +     |
// 40 |       >       |       =       |   Unordered   |    -     |
// 41 |       >       |       >       |       <       |    -     |
// 42 |       >       |       >       |       =       |    -     |
// 43 |       >       |       >       |       >       |    -?    |
// 44 |       >       |       >       |   Unordered   |    -     |
// 45 |       >       |   Unordered   |       <       |    -     |
// 46 |       >       |   Unordered   |       =       |    -     |
// 47 |       >       |   Unordered   |       >       |    -?    |
// 48 |       >       |   Unordered   |   Unordered   |    -?    |
// 49 |   Unordered   |       <       |       <       |    -?    |
// 50 |   Unordered   |       <       |       =       |    -     |
// 51 |   Unordered   |       <       |       >       |    -     |
// 52 |   Unordered   |       <       |   Unordered   |    +     |
// 53 |   Unordered   |       =       |       <       |    -     |
// 54 |   Unordered   |       =       |       =       |    -     |
// 55 |   Unordered   |       =       |       >       |    -     |
// 56 |   Unordered   |       =       |   Unordered   |    +     |
// 57 |   Unordered   |       >       |       <       |    -     |
// 58 |   Unordered   |       >       |       =       |    -     |
// 59 |   Unordered   |       >       |       >       |    +     |
// 60 |   Unordered   |       >       |   Unordered   |    +     |
// 61 |   Unordered   |   Unordered   |       <       |    +?    |
// 62 |   Unordered   |   Unordered   |       =       |    +?    |
// 63 |   Unordered   |   Unordered   |       >       |    +?    |
// 64 |   Unordered   |   Unordered   |   Unordered   |    +     |

#![feature(decl_macro, rustc_attrs)]

struct Right;
// struct Wrong; // not defined

macro_rules! include { () => {
    macro_rules! gen_outer { () => {
        macro_rules! m { () => { Wrong } }
    }}
    macro_rules! gen_inner { () => {
        macro_rules! m { () => { Right } }
    }}
    macro_rules! gen_invoc { () => {
        m!()
    }}

    // -----------------------------------------------------------

    fn check1() {
        macro_rules! m { () => {} }

        macro_rules! gen_gen_inner_invoc { () => {
            gen_inner!();
            m!(); //~ ERROR `m` is ambiguous
        }}
        gen_gen_inner_invoc!();
    }

    fn check5() {
        macro_rules! m { () => { Wrong } }

        macro_rules! gen_inner_invoc { () => {
            macro_rules! m { () => { Right } }
            m!(); // OK
        }}
        gen_inner_invoc!();
    }

    fn check9() {
        macro_rules! m { () => { Wrong } }

        macro_rules! gen_inner_gen_invoc { () => {
            macro_rules! m { () => { Right } }
            gen_invoc!(); // OK
        }}
        gen_inner_gen_invoc!();
    }

    fn check10() {
        macro_rules! m { () => { Wrong } }

        macro_rules! m { () => { Right } }

        gen_invoc!(); // OK
    }

    fn check13() {
        macro_rules! m { () => {} }

        gen_inner!();

        macro_rules! gen_invoc { () => { m!() } } //~ ERROR `m` is ambiguous
        gen_invoc!();
    }

    fn check17() {
        macro_rules! m { () => {} }

        gen_inner!();

        m!(); //~ ERROR `m` is ambiguous
    }

    fn check22() {
        macro_rules! m { () => { Wrong } }

        macro_rules! m { () => { Right } }

        m!(); // OK
    }

    fn check36() {
        gen_outer!();

        gen_inner!();

        m!(); //~ ERROR `m` is ambiguous
    }

    fn check39() {
        gen_outer!();

        macro_rules! m { () => { Right } }

        m!(); // OK
    }

    fn check52() {
        gen_outer!();

        macro_rules! gen_gen_inner_invoc { () => {
            gen_inner!();
            m!(); //~ ERROR `m` is ambiguous
        }}
        gen_gen_inner_invoc!();
    }

    fn check56() {
        gen_outer!();

        macro_rules! gen_inner_invoc { () => {
            macro_rules! m { () => { Right } }
            m!(); // OK
        }}
        gen_inner_invoc!();
    }

    fn check59() {
        gen_outer!();

        macro_rules! m { () => { Right } }

        gen_invoc!(); // OK
    }

    fn check60() {
        gen_outer!();

        macro_rules! gen_inner_gen_invoc { () => {
            macro_rules! m { () => { Right } }
            gen_invoc!(); // OK
        }}
        gen_inner_gen_invoc!();
    }

    fn check64() {
        gen_outer!();

        gen_inner!();

        macro_rules! gen_invoc { () => { m!() } } //~ ERROR `m` is ambiguous
        gen_invoc!();
    }

    // -----------------------------------------------------------
    // These configurations are only possible with legacy macro scoping

    fn check33() {
        macro_rules! gen_outer_gen_inner { () => {
            macro_rules! m { () => {} }
            gen_inner!();
        }}
        gen_outer_gen_inner!();

        m!(); //~ ERROR `m` is ambiguous
    }

    fn check34() {
        macro_rules! gen_outer_inner { () => {
            macro_rules! m { () => { Wrong } }
            macro_rules! m { () => { Right } }
        }}
        gen_outer_inner!();

        m!(); // OK
    }

    fn check35() {
        macro_rules! gen_gen_outer_inner { () => {
            gen_outer!();
            macro_rules! m { () => { Right } }
        }}
        gen_gen_outer_inner!();

        m!(); // OK
    }

    fn check61() {
        macro_rules! gen_outer_gen_inner { () => {
            macro_rules! m { () => {} }
            gen_inner!();
        }}
        gen_outer_gen_inner!();

        macro_rules! gen_invoc { () => { m!() } } //~ ERROR `m` is ambiguous
        gen_invoc!();
    }

    fn check62() {
        macro_rules! gen_outer_inner { () => {
            macro_rules! m { () => { Wrong } }
            macro_rules! m { () => { Right } }
        }}
        gen_outer_inner!();

        gen_invoc!(); // OK
    }

    fn check63() {
        macro_rules! gen_gen_outer_inner { () => {
            gen_outer!();
            macro_rules! m { () => { Right } }
        }}
        gen_gen_outer_inner!();

        gen_invoc!(); // OK
    }
}}

include!();

fn main() {}
