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
// 33 |       >       |       <       |       <       |    -?    |
// 34 |       >       |       <       |       =       |    -?    |
// 35 |       >       |       <       |       >       |    -?    |
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
// 61 |   Unordered   |   Unordered   |       <       |    -?    |
// 62 |   Unordered   |   Unordered   |       =       |    -?    |
// 63 |   Unordered   |   Unordered   |       >       |    -?    |
// 64 |   Unordered   |   Unordered   |   Unordered   |    +     |

#![feature(decl_macro, rustc_attrs)]

struct Right;
// struct Wrong; // not defined

#[rustc_transparent_macro]
macro include() {
    #[rustc_transparent_macro]
    macro gen_outer() {
        macro m() { Wrong }
    }
    #[rustc_transparent_macro]
    macro gen_inner() {
        macro m() { Right }
    }
    #[rustc_transparent_macro]
    macro gen_invoc() {
        m!()
    }

    // -----------------------------------------------------------

    fn check1() {
        macro m() {}
        {
            #[rustc_transparent_macro]
            macro gen_gen_inner_invoc() {
                gen_inner!();
                m!(); //~ ERROR `m` is ambiguous
            }
            gen_gen_inner_invoc!();
        }
    }

    fn check5() {
        macro m() { Wrong }
        {
            #[rustc_transparent_macro]
            macro gen_inner_invoc() {
                macro m() { Right }
                m!(); // OK
            }
            gen_inner_invoc!();
        }
    }

    fn check9() {
        macro m() { Wrong }
        {
            #[rustc_transparent_macro]
            macro gen_inner_gen_invoc() {
                macro m() { Right }
                gen_invoc!(); // OK
            }
            gen_inner_gen_invoc!();
        }
    }

    fn check10() {
        macro m() { Wrong }
        {
            macro m() { Right }
            gen_invoc!(); // OK
        }
    }

    fn check13() {
        macro m() {}
        {
            gen_inner!();
            #[rustc_transparent_macro]
            macro gen_invoc() { m!() } //~ ERROR `m` is ambiguous
            gen_invoc!();
        }
    }

    fn check17() {
        macro m() {}
        {
            gen_inner!();
            m!(); //~ ERROR `m` is ambiguous
        }
    }

    fn check22() {
        macro m() { Wrong }
        {
            macro m() { Right }
            m!(); // OK
        }
    }

    fn check36() {
        gen_outer!();
        {
            gen_inner!();
            m!(); //~ ERROR `m` is ambiguous
        }
    }

    fn check39() {
        gen_outer!();
        {
            macro m() { Right }
            m!(); // OK
        }
    }

    fn check52() {
        gen_outer!();
        {
            #[rustc_transparent_macro]
            macro gen_gen_inner_invoc() {
                gen_inner!();
                m!(); //~ ERROR `m` is ambiguous
            }
            gen_gen_inner_invoc!();
        }
    }

    fn check56() {
        gen_outer!();
        {
            #[rustc_transparent_macro]
            macro gen_inner_invoc() {
                macro m() { Right }
                m!(); // OK
            }
            gen_inner_invoc!();
        }
    }

    fn check59() {
        gen_outer!();
        {
            macro m() { Right }
            gen_invoc!(); // OK
        }
    }

    fn check60() {
        gen_outer!();
        {
            #[rustc_transparent_macro]
            macro gen_inner_gen_invoc() {
                macro m() { Right }
                gen_invoc!(); // OK
            }
            gen_inner_gen_invoc!();
        }
    }

    fn check64() {
        gen_outer!();
        {
            gen_inner!();
            #[rustc_transparent_macro]
            macro gen_invoc() { m!() } //~ ERROR `m` is ambiguous
            gen_invoc!();
        }
    }
}

include!();

fn main() {}
