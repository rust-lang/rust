// Ensure that generic parameters always have modern hygiene.

//@ check-pass

#![feature(decl_macro, rustc_attrs)]

mod type_params {
    macro m($T:ident) {
        fn f<$T: Clone, T: PartialEq>(t1: $T, t2: T) -> ($T, bool) {
            (t1.clone(), t2 == t2)
        }
    }

    #[rustc_macro_transparency = "semitransparent"]
    macro n($T:ident) {
        fn g<$T: Clone>(t1: $T, t2: T) -> (T, $T) {
            (t1.clone(), t2.clone())
        }
        fn h<T: Clone>(t1: $T, t2: T) -> (T, $T) {
            (t1.clone(), t2.clone())
        }
    }

    #[rustc_macro_transparency = "transparent"]
    macro p($T:ident) {
        fn j<$T: Clone>(t1: $T, t2: T) -> (T, $T) {
            (t1.clone(), t2.clone())
        }
        fn k<T: Clone>(t1: $T, t2: T) -> (T, $T) {
            (t1.clone(), t2.clone())
        }
    }

    m!(T);
    n!(T);
    p!(T);
}

mod lifetime_params {
    macro m($a:lifetime) {
        fn f<'b, 'c, $a: 'b, 'a: 'c>(t1: &$a(), t2: &'a ()) -> (&'b (), &'c ()) {
            (t1, t2)
        }
    }

    #[rustc_macro_transparency = "semitransparent"]
    macro n($a:lifetime) {
        fn g<$a>(t1: &$a(), t2: &'a ()) -> (&'a (), &$a ()) {
            (t1, t2)
        }
        fn h<'a>(t1: &$a(), t2: &'a ()) -> (&'a (), &$a ()) {
            (t1, t2)
        }
    }

    #[rustc_macro_transparency = "transparent"]
    macro p($a:lifetime) {
        fn j<$a>(t1: &$a(), t2: &'a ()) -> (&'a (), &$a ()) {
            (t1, t2)
        }
        fn k<'a>(t1: &$a(), t2: &'a ()) -> (&'a (), &$a ()) {
            (t1, t2)
        }
    }

    m!('a);
    n!('a);
    p!('a);
}

mod const_params {
    macro m($C:ident) {
        fn f<const $C: usize, const C: usize>(t1: [(); $C], t2: [(); C]) -> ([(); $C], [(); C]) {
            (t1, t2)
        }
    }

    #[rustc_macro_transparency = "semitransparent"]
    macro n($C:ident) {
        fn g<const $C: usize>(t1: [(); $C], t2: [(); C]) -> ([(); C], [(); $C]) {
            (t1, t2)
        }
        fn h<const C: usize>(t1: [(); $C], t2: [(); C]) -> ([(); C], [(); $C]) {
            (t1, t2)
        }
    }

    #[rustc_macro_transparency = "transparent"]
    macro p($C:ident) {
        fn j<const $C: usize>(t1: [(); $C], t2: [(); C]) -> ([(); C], [(); $C]) {
            (t1, t2)
        }
        fn k<const C: usize>(t1: [(); $C], t2: [(); C]) -> ([(); C], [(); $C]) {
            (t1, t2)
        }
    }

    m!(C);
    n!(C);
    p!(C);
}

fn main() {}
