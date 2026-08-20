//@ known-bug: #148095
use std::ops::Mul;

struct Quantity<S>(S);
impl<S> Mul<Quantity<<f32 as Mul<S>>::Output>> for f32
where
    Quantity<Self::Output>:
{
    type Output = ();
    fn mul(self, _: Quantity<<f32 as Mul<S>>::Output>) {}
}
