// ignore-tidy-linelength

//@ run-pass
#![allow(dead_code)]

// There are five cfg's below. I explored the set of all non-empty combinations
// of the below five cfg's, which is 2^5 - 1 = 31 combinations.
//
// Of the 31, 11 resulted in ambiguous method resolutions; while it may be good
// to have a test for all of the eleven variations of that error, I am not sure
// this particular test is the best way to encode it. So they are skipped in
// this revisions list (but not in the expansion mapping the binary encoding to
// the corresponding cfg flags).
//
// Notable, here are the cases that will be incompatible if something does not override them first:
// {bar_for_foo, valbar_for_et_foo}: these are higher precedent than the `&mut self` method on `Foo`, and so no case matching bx1x1x is included.
// {mutbar_for_foo, valbar_for_etmut_foo} (which are lower precedent than the inherent `&mut self` method on `Foo`; e.g. b10101 *is* included.

//@ revisions: b00001 b00010 b00011 b00100 b00101 b00110 b00111 b01000 b01001 b01100 b01101 b10000 b10001 b10010 b10011 b10101 b10111 b11000 b11001 b11101
//@ unused-revision-names: b01010 b01011 b01110 b01111 b10100 b10110 b11010 b11011 b11100 b11110 b11111

//@ compile-flags: --check-cfg=cfg(inherent_mut,bar_for_foo,mutbar_for_foo)
//@ compile-flags: --check-cfg=cfg(valbar_for_et_foo,valbar_for_etmut_foo)

//@[b00001]compile-flags:  --cfg inherent_mut
//@[b00010]compile-flags:                     --cfg bar_for_foo
//@[b00011]compile-flags:  --cfg inherent_mut --cfg bar_for_foo
//@[b00100]compile-flags:                                       --cfg mutbar_for_foo
//@[b00101]compile-flags:  --cfg inherent_mut                   --cfg mutbar_for_foo
//@[b00110]compile-flags:                     --cfg bar_for_foo --cfg mutbar_for_foo
//@[b00111]compile-flags:  --cfg inherent_mut --cfg bar_for_foo --cfg mutbar_for_foo
//@[b01000]compile-flags:                                                            --cfg valbar_for_et_foo
//@[b01001]compile-flags:  --cfg inherent_mut                                        --cfg valbar_for_et_foo
//@[b01010]compile-flags:                     --cfg bar_for_foo                      --cfg valbar_for_et_foo
//@[b01011]compile-flags:  --cfg inherent_mut --cfg bar_for_foo                      --cfg valbar_for_et_foo
//@[b01100]compile-flags:                                       --cfg mutbar_for_foo --cfg valbar_for_et_foo
//@[b01101]compile-flags:  --cfg inherent_mut                   --cfg mutbar_for_foo --cfg valbar_for_et_foo
//@[b01110]compile-flags:                     --cfg bar_for_foo --cfg mutbar_for_foo --cfg valbar_for_et_foo
//@[b01111]compile-flags:  --cfg inherent_mut --cfg bar_for_foo --cfg mutbar_for_foo --cfg valbar_for_et_foo
//@[b10000]compile-flags:                                                                                    --cfg valbar_for_etmut_foo
//@[b10001]compile-flags:  --cfg inherent_mut                                                                --cfg valbar_for_etmut_foo
//@[b10010]compile-flags:                     --cfg bar_for_foo                                              --cfg valbar_for_etmut_foo
//@[b10011]compile-flags:  --cfg inherent_mut --cfg bar_for_foo                                              --cfg valbar_for_etmut_foo
//@[b10100]compile-flags:                                       --cfg mutbar_for_foo                         --cfg valbar_for_etmut_foo
//@[b10101]compile-flags:  --cfg inherent_mut                   --cfg mutbar_for_foo                         --cfg valbar_for_etmut_foo
//@[b10110]compile-flags:                     --cfg bar_for_foo --cfg mutbar_for_foo                         --cfg valbar_for_etmut_foo
//@[b10111]compile-flags:  --cfg inherent_mut --cfg bar_for_foo --cfg mutbar_for_foo                         --cfg valbar_for_etmut_foo
//@[b11000]compile-flags:                                                            --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11001]compile-flags:  --cfg inherent_mut                                        --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11010]compile-flags:                     --cfg bar_for_foo                      --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11011]compile-flags:  --cfg inherent_mut --cfg bar_for_foo                      --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11100]compile-flags:                                       --cfg mutbar_for_foo --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11101]compile-flags:  --cfg inherent_mut                   --cfg mutbar_for_foo --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11110]compile-flags:                     --cfg bar_for_foo --cfg mutbar_for_foo --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo
//@[b11111]compile-flags:  --cfg inherent_mut --cfg bar_for_foo --cfg mutbar_for_foo --cfg valbar_for_et_foo --cfg valbar_for_etmut_foo

struct Foo {}

type S = &'static str;

trait Bar {
    fn bar(&self, _: &str) -> S;
}

trait MutBar {
    fn bar(&mut self, _: &str) -> S;
}

trait ValBar {
    fn bar(self, _: &str) -> S;
}

#[cfg(inherent_mut)]
impl Foo {
    fn bar(&mut self, _: &str) -> S {
        "In struct impl!"
    }
}

#[cfg(bar_for_foo)]
impl Bar for Foo {
    fn bar(&self, _: &str) -> S {
        "In trait &self impl!"
    }
}

#[cfg(mutbar_for_foo)]
impl MutBar for Foo {
    fn bar(&mut self, _: &str) -> S {
        "In trait &mut self impl!"
    }
}

#[cfg(valbar_for_et_foo)]
impl ValBar for &Foo {
    fn bar(self, _: &str) -> S {
        "In trait self impl for &Foo!"
    }
}

#[cfg(valbar_for_etmut_foo)]
impl ValBar for &mut Foo {
    fn bar(self, _: &str) -> S {
        "In trait self impl for &mut Foo!"
    }
}

fn main() {
    #![allow(unused_mut)] // some of the impls above will want it.

    #![allow(unreachable_patterns)] // the cfg-coding pattern below generates unreachable patterns.

    {
        macro_rules! all_variants_on_value {
            ($e:expr) => {
                match $e {
                    #[cfg(bar_for_foo)]
                    x => assert_eq!(x, "In trait &self impl!"),

                    #[cfg(valbar_for_et_foo)]
                    x => assert_eq!(x, "In trait self impl for &Foo!"),

                    #[cfg(inherent_mut)]
                    x => assert_eq!(x, "In struct impl!"),

                    #[cfg(mutbar_for_foo)]
                    x => assert_eq!(x, "In trait &mut self impl!"),

                    #[cfg(valbar_for_etmut_foo)]
                    x => assert_eq!(x, "In trait self impl for &mut Foo!"),
                }
            }
        }

        let mut f = Foo {};
        all_variants_on_value!(f.bar("f.bar"));

        let f_mr = &mut Foo {};
        all_variants_on_value!((*f_mr).bar("(*f_mr).bar"));
    }

    // This is sort of interesting: `&mut Foo` ends up with a significantly
    // different resolution order than what was devised above. Presumably this
    // is because we can get to a `&self` method by first a deref of the given
    // `&mut Foo` and then an autoref, and that is a longer path than a mere
    // auto-ref of a `Foo`.

    {
        let f_mr = &mut Foo {};

        match f_mr.bar("f_mr.bar") {
            #[cfg(inherent_mut)]
            x => assert_eq!(x, "In struct impl!"),

            #[cfg(valbar_for_etmut_foo)]
            x => assert_eq!(x, "In trait self impl for &mut Foo!"),

            #[cfg(mutbar_for_foo)]
            x => assert_eq!(x, "In trait &mut self impl!"),

            #[cfg(valbar_for_et_foo)]
            x => assert_eq!(x, "In trait self impl for &Foo!"),

            #[cfg(bar_for_foo)]
            x => assert_eq!(x, "In trait &self impl!"),
        }
    }


    // Note that this isn't actually testing a resolution order; if both of these are
    // enabled, it yields an ambiguous method resolution error. The test tries to embed
    // that fact by testing *both* orders (and so the only way that can be right is if
    // they are not actually compatible).
    #[cfg(any(bar_for_foo, valbar_for_et_foo))]
    {
        let f_r = &Foo {};

        match f_r.bar("f_r.bar") {
            #[cfg(bar_for_foo)]
            x => assert_eq!(x, "In trait &self impl!"),

            #[cfg(valbar_for_et_foo)]
            x => assert_eq!(x, "In trait self impl for &Foo!"),
        }

        match f_r.bar("f_r.bar") {
            #[cfg(valbar_for_et_foo)]
            x => assert_eq!(x, "In trait self impl for &Foo!"),

            #[cfg(bar_for_foo)]
            x => assert_eq!(x, "In trait &self impl!"),
        }
    }

}
