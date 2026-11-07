pub struct Closed01<F>(pub F);

pub trait Bar { fn new() -> Self; }

impl<T: Bar> Bar for Closed01<T> {
    fn new() -> Closed01<T> { Closed01(Bar::new()) }
}
impl Bar for f32 { fn new() -> f32 { 1.0 } }

pub fn random<T: Bar>() -> T { Bar::new() }
