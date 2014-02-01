// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! tuple_extensions {
    ($(
        ($move_trait:ident, $immutable_trait:ident) {
            $(($get_fn:ident, $get_ref_fn:ident) -> $T:ident {
                $move_pattern:pat, $ref_pattern:pat => $ret:expr
            })+
        }
    )+) => {
        $(
            pub trait $move_trait<$($T),+> {
                $(fn $get_fn(self) -> $T;)+
            }

            impl<$($T),+> $move_trait<$($T),+> for ($($T,)+) {
                $(
                    #[inline]
                    fn $get_fn(self) -> $T {
                        let $move_pattern = self;
                        $ret
                    }
                )+
            }

            pub trait $immutable_trait<$($T),+> {
                $(fn $get_ref_fn<'a>(&'a self) -> &'a $T;)+
            }

            impl<$($T),+> $immutable_trait<$($T),+> for ($($T,)+) {
                $(
                    #[inline]
                    fn $get_ref_fn<'a>(&'a self) -> &'a $T {
                        let $ref_pattern = *self;
                        $ret
                    }
                )+
            }
        )+
    }

}

tuple_extensions! {
    (Tuple1, ImmutableTuple1) {
        (n0, n0_ref) -> A { (a,), (ref a,) => a }
    }

    (Tuple2, ImmutableTuple2) {
        (n0, n0_ref) -> A { (a,_), (ref a,_) => a }
        (n1, n1_ref) -> B { (_,b), (_,ref b) => b }
    }

    (Tuple3, ImmutableTuple3) {
        (n0, n0_ref) -> A { (a,_,_), (ref a,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_), (_,ref b,_) => b }
        (n2, n2_ref) -> C { (_,_,c), (_,_,ref c) => c }
    }

    (Tuple4, ImmutableTuple4) {
        (n0, n0_ref) -> A { (a,_,_,_), (ref a,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_), (_,ref b,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_), (_,_,ref c,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d), (_,_,_,ref d) => d }
    }

    (Tuple5, ImmutableTuple5) {
        (n0, n0_ref) -> A { (a,_,_,_,_), (ref a,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_), (_,ref b,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_), (_,_,ref c,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_), (_,_,_,ref d,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e), (_,_,_,_,ref e) => e }
    }

    (Tuple6, ImmutableTuple6) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_), (ref a,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_), (_,ref b,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_), (_,_,ref c,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_), (_,_,_,ref d,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_), (_,_,_,_,ref e,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f), (_,_,_,_,_,ref f) => f }
    }

    (Tuple7, ImmutableTuple7) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_), (ref a,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_), (_,ref b,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_), (_,_,ref c,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_), (_,_,_,ref d,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_), (_,_,_,_,ref e,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_), (_,_,_,_,_,ref f,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g), (_,_,_,_,_,_,ref g) => g }
    }

    (Tuple8, ImmutableTuple8) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_), (_,_,ref c,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_), (_,_,_,ref d,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_), (_,_,_,_,ref e,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_), (_,_,_,_,_,ref f,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_), (_,_,_,_,_,_,ref g,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h), (_,_,_,_,_,_,_,ref h) => h }
    }

    (Tuple9, ImmutableTuple9) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_,_), (_,_,_,_,ref e,_,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_,_), (_,_,_,_,_,ref f,_,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_,_), (_,_,_,_,_,_,ref g,_,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h,_), (_,_,_,_,_,_,_,ref h,_) => h }
        (n8, n8_ref) -> I { (_,_,_,_,_,_,_,_,i), (_,_,_,_,_,_,_,_,ref i) => i }
    }

    (Tuple10, ImmutableTuple10) {
        (n0, n0_ref) -> A { (a,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_) => a }
        (n1, n1_ref) -> B { (_,b,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_) => b }
        (n2, n2_ref) -> C { (_,_,c,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_) => c }
        (n3, n3_ref) -> D { (_,_,_,d,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_) => d }
        (n4, n4_ref) -> E { (_,_,_,_,e,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_) => e }
        (n5, n5_ref) -> F { (_,_,_,_,_,f,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_) => f }
        (n6, n6_ref) -> G { (_,_,_,_,_,_,g,_,_,_), (_,_,_,_,_,_,ref g,_,_,_) => g }
        (n7, n7_ref) -> H { (_,_,_,_,_,_,_,h,_,_), (_,_,_,_,_,_,_,ref h,_,_) => h }
        (n8, n8_ref) -> I { (_,_,_,_,_,_,_,_,i,_), (_,_,_,_,_,_,_,_,ref i,_) => i }
        (n9, n9_ref) -> J { (_,_,_,_,_,_,_,_,_,j), (_,_,_,_,_,_,_,_,_,ref j) => j }
    }

    (Tuple11, ImmutableTuple11) {
        (n0,  n0_ref)  -> A { (a,_,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_,_) => a }
        (n1,  n1_ref)  -> B { (_,b,_,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_,_) => b }
        (n2,  n2_ref)  -> C { (_,_,c,_,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_,_) => c }
        (n3,  n3_ref)  -> D { (_,_,_,d,_,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_,_) => d }
        (n4,  n4_ref)  -> E { (_,_,_,_,e,_,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_,_) => e }
        (n5,  n5_ref)  -> F { (_,_,_,_,_,f,_,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_,_) => f }
        (n6,  n6_ref)  -> G { (_,_,_,_,_,_,g,_,_,_,_), (_,_,_,_,_,_,ref g,_,_,_,_) => g }
        (n7,  n7_ref)  -> H { (_,_,_,_,_,_,_,h,_,_,_), (_,_,_,_,_,_,_,ref h,_,_,_) => h }
        (n8,  n8_ref)  -> I { (_,_,_,_,_,_,_,_,i,_,_), (_,_,_,_,_,_,_,_,ref i,_,_) => i }
        (n9,  n9_ref)  -> J { (_,_,_,_,_,_,_,_,_,j,_), (_,_,_,_,_,_,_,_,_,ref j,_) => j }
        (n10, n10_ref) -> K { (_,_,_,_,_,_,_,_,_,_,k), (_,_,_,_,_,_,_,_,_,_,ref k) => k }
    }

    (Tuple12, ImmutableTuple12) {
        (n0,  n0_ref)  -> A { (a,_,_,_,_,_,_,_,_,_,_,_), (ref a,_,_,_,_,_,_,_,_,_,_,_) => a }
        (n1,  n1_ref)  -> B { (_,b,_,_,_,_,_,_,_,_,_,_), (_,ref b,_,_,_,_,_,_,_,_,_,_) => b }
        (n2,  n2_ref)  -> C { (_,_,c,_,_,_,_,_,_,_,_,_), (_,_,ref c,_,_,_,_,_,_,_,_,_) => c }
        (n3,  n3_ref)  -> D { (_,_,_,d,_,_,_,_,_,_,_,_), (_,_,_,ref d,_,_,_,_,_,_,_,_) => d }
        (n4,  n4_ref)  -> E { (_,_,_,_,e,_,_,_,_,_,_,_), (_,_,_,_,ref e,_,_,_,_,_,_,_) => e }
        (n5,  n5_ref)  -> F { (_,_,_,_,_,f,_,_,_,_,_,_), (_,_,_,_,_,ref f,_,_,_,_,_,_) => f }
        (n6,  n6_ref)  -> G { (_,_,_,_,_,_,g,_,_,_,_,_), (_,_,_,_,_,_,ref g,_,_,_,_,_) => g }
        (n7,  n7_ref)  -> H { (_,_,_,_,_,_,_,h,_,_,_,_), (_,_,_,_,_,_,_,ref h,_,_,_,_) => h }
        (n8,  n8_ref)  -> I { (_,_,_,_,_,_,_,_,i,_,_,_), (_,_,_,_,_,_,_,_,ref i,_,_,_) => i }
        (n9,  n9_ref)  -> J { (_,_,_,_,_,_,_,_,_,j,_,_), (_,_,_,_,_,_,_,_,_,ref j,_,_) => j }
        (n10, n10_ref) -> K { (_,_,_,_,_,_,_,_,_,_,k,_), (_,_,_,_,_,_,_,_,_,_,ref k,_) => k }
        (n11, n11_ref) -> L { (_,_,_,_,_,_,_,_,_,_,_,l), (_,_,_,_,_,_,_,_,_,_,_,ref l) => l }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_n_tuple() {
        let t = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
        assert_eq!(t.n0(), 0u8);
        assert_eq!(t.n1(), 1u16);
        assert_eq!(t.n2(), 2u32);
        assert_eq!(t.n3(), 3u64);
        assert_eq!(t.n4(), 4u);
        assert_eq!(t.n5(), 5i8);
        assert_eq!(t.n6(), 6i16);
        assert_eq!(t.n7(), 7i32);
        assert_eq!(t.n8(), 8i64);
        assert_eq!(t.n9(), 9i);
        assert_eq!(t.n10(), 10f32);
        assert_eq!(t.n11(), 11f64);

        assert_eq!(t.n0_ref(), &0u8);
        assert_eq!(t.n1_ref(), &1u16);
        assert_eq!(t.n2_ref(), &2u32);
        assert_eq!(t.n3_ref(), &3u64);
        assert_eq!(t.n4_ref(), &4u);
        assert_eq!(t.n5_ref(), &5i8);
        assert_eq!(t.n6_ref(), &6i16);
        assert_eq!(t.n7_ref(), &7i32);
        assert_eq!(t.n8_ref(), &8i64);
        assert_eq!(t.n9_ref(), &9i);
        assert_eq!(t.n10_ref(), &10f32);
        assert_eq!(t.n11_ref(), &11f64);
    }
}