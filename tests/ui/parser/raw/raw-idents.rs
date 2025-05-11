//@ check-pass
//@ revisions:e2015 e2018 e2021 e2024
//@[e2015] edition:2015
//@[e2018] edition:2018
//@[e2021] edition:2021
//@[e2024] edition:2024

// Ensure that all (usable as identifier) keywords work as raw identifiers in all positions.
// This was motivated by issue #137128, where `r#move`/`r#static`` did not work as `const` names
// due to a parser check not acounting for raw identifiers.

#![crate_type = "lib"]
#![allow(dead_code, nonstandard_style)]

// NOTE: It is vital to only use a `tt` fragment to avoid confusing
// the parser with nonterminals that can mask bugs.

macro_rules! tests {
    ($kw:tt) => {
        mod $kw {
            mod const_item {
                const $kw: () = ();
            }
            mod static_item {
                static $kw: () = ();
            }
            mod fn_item {
                fn $kw() {}
            }
            mod mod_and_use_item {
                mod $kw {
                    use super::$kw;
                }
            }
            mod ty_alias_item {
                type $kw = ();
            }
            mod struct_item {
                struct $kw { $kw: () }
            }
            mod enum_item {
                enum $kw { $kw }
            }
            mod union_item {
                union $kw { $kw: () }
            }
            mod trait_item {
                trait $kw {
                    fn $kw() {}
                }
            }
            mod generics_and_impl {
                struct A<$kw>($kw);
                enum B<$kw> { A($kw) }
                trait Tr<$kw> {
                    type $kw;
                }

                impl<$kw> Tr<$kw> for A<$kw> {
                    type $kw = ();
                }
                impl<$kw> B<$kw> {}
            }
            mod extern_crate {
                #[cfg(any())]
                extern crate $kw;
            }
            mod body {
                fn expr() {
                    let $kw = 0;
                    let b = $kw;
                    assert_eq!($kw, b);
                    type $kw = ();
                    let $kw: $kw = ();
                    let _ = $kw as $kw;
                }
                fn pat_const() {
                    const $kw: u8 = 0;

                    // Ensure that $kw actually matches the constant.
                    #[forbid(unreachable_patterns)]
                    match 1 {
                        $kw => {}
                        _ => {}
                    }
                }
                fn pat_binding() {
                    match 1 {
                        $kw => {}
                        _ => {}
                    }
                }
            }
        }
    };
}

tests!(r#break);
tests!(r#const);
tests!(r#continue);
tests!(r#else);
tests!(r#enum);
tests!(r#extern);
tests!(r#false);
tests!(r#fn);
tests!(r#for);
tests!(r#if);
tests!(r#impl);
tests!(r#in);
tests!(r#let);
tests!(r#loop);
tests!(r#match);
tests!(r#mod);
tests!(r#move);
tests!(r#mut);
tests!(r#pub);
tests!(r#ref);
tests!(r#return);
tests!(r#static);
tests!(r#struct);
tests!(r#trait);
tests!(r#true);
tests!(r#type);
tests!(r#unsafe);
tests!(r#use);
tests!(r#where);
tests!(r#while);
tests!(r#abstract);
tests!(r#become);
tests!(r#box);
tests!(r#do);
tests!(r#final);
tests!(r#macro);
tests!(r#override);
tests!(r#priv);
tests!(r#typeof);
tests!(r#unsized);
tests!(r#virtual);
tests!(r#yield);
tests!(r#async);
tests!(r#await);
tests!(r#dyn);
tests!(r#gen);
tests!(r#try);

// Weak keywords:
tests!(auto);
tests!(builtin);
tests!(catch);
tests!(default);
tests!(macro_rules);
tests!(raw);
tests!(reuse);
tests!(contract_ensures);
tests!(contract_requires);
tests!(safe);
tests!(union);
tests!(yeet);
