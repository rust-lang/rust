// check-pass

mod convenience_operators {
    use crate::{Op, Relation};
    use std::ops::AddAssign;
    use std::ops::Mul;

    impl<C: Op> Relation<C> {
        pub fn map<F: Fn(C::D) -> D2 + 'static, D2: 'static>(
            self,
            f: F,
        ) -> Relation<impl Op<D = D2, R = C::R>> {
            self.map_dr(move |x, r| (f(x), r))
        }
    }

    impl<K: 'static, V: 'static, C: Op<D = (K, V)>> Relation<C> {
        pub fn semijoin<C2: Op<D = K, R = R2>, R2, R3: AddAssign<R3>>(
            self,
            other: Relation<C2>,
        ) -> Relation<impl Op<D = C::D, R = R3>>
        where
            C::R: Mul<R2, Output = R3>,
        {
            self.join(other.map(|x| (x, ()))).map(|(k, x, ())| (k, x))
        }
    }
}

mod core {
    mod operator {
        mod join {
            use super::Op;
            use crate::core::Relation;
            use std::ops::{AddAssign, Mul};
            struct Join<LC, RC> {
                _left: LC,
                _right: RC,
            }
            impl<
                    LC: Op<D = (K, LD), R = LR>,
                    RC: Op<D = (K, RD), R = RR>,
                    K: 'static,
                    LD: 'static,
                    LR: AddAssign<LR> + Mul<RR, Output = OR>,
                    RD: 'static,
                    RR: AddAssign<RR>,
                    OR: AddAssign<OR>,
                > Op for Join<LC, RC>
            {
                type D = (K, LD, RD);
                type R = OR;
            }
            impl<K: 'static, D: 'static, C: Op<D = (K, D)>> Relation<C> {
                pub fn join<C2: Op<D = (K, D2)>, D2: 'static, OR: AddAssign<OR>>(
                    self,
                    other: Relation<C2>,
                ) -> Relation<impl Op<D = (K, D, D2), R = OR>>
                where
                    C::R: Mul<C2::R, Output = OR>,
                {
                    Relation {
                        inner: Join {
                            _left: self.inner,
                            _right: other.inner,
                        },
                    }
                }
            }
        }
        mod map {
            use super::Op;
            use crate::core::Relation;
            use std::ops::AddAssign;
            struct Map<C, MF> {
                _inner: C,
                _op: MF,
            }
            impl<
                    D1,
                    R1,
                    D2: 'static,
                    R2: AddAssign<R2>,
                    C: Op<D = D1, R = R1>,
                    MF: Fn(D1, R1) -> (D2, R2),
                > Op for Map<C, MF>
            {
                type D = D2;
                type R = R2;
            }
            impl<C: Op> Relation<C> {
                pub fn map_dr<F: Fn(C::D, C::R) -> (D2, R2), D2: 'static, R2: AddAssign<R2>>(
                    self,
                    f: F,
                ) -> Relation<impl Op<D = D2, R = R2>> {
                    Relation {
                        inner: Map {
                            _inner: self.inner,
                            _op: f,
                        },
                    }
                }
            }
        }
        use std::ops::AddAssign;
        pub trait Op {
            type D: 'static;
            type R: AddAssign<Self::R>;
        }
    }
    pub use self::operator::Op;
    #[derive(Clone)]
    pub struct Relation<C> {
        inner: C,
    }
}

use self::core::Op;
pub use self::core::Relation;

fn main() {}
