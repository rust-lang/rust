#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait TensorDimension {
    const DIM: usize;
    const ISSCALAR: bool = Self::DIM == 0;
    fn is_scalar(&self) -> bool {
        Self::ISSCALAR
    }
}

trait TensorSize: TensorDimension {
    fn size(&self) -> [usize; Self::DIM];
    fn inbounds(&self, index: [usize; Self::DIM]) -> bool {
        index.iter().zip(self.size().iter()).all(|(i, s)| i < s)
    }
}

trait Broadcastable: TensorSize + Sized {
    type Element;
    fn bget(&self, index: [usize; Self::DIM]) -> Option<Self::Element>;
    fn lazy_updim<const NEWDIM: usize>(
        &self,
        size: [usize; NEWDIM],
    ) -> LazyUpdim<Self, { Self::DIM }, NEWDIM> {
        assert!(
            NEWDIM >= Self::DIM,
            "Updimmed tensor cannot have fewer indices than the initial one."
        );
        LazyUpdim { size, reference: &self }
    }
    fn bmap<T, F: Fn(Self::Element) -> T>(&self, foo: F) -> BMap<T, Self, F, { Self::DIM }> {
        BMap { reference: self, closure: foo }
    }
}

struct LazyUpdim<'a, T: Broadcastable, const OLDDIM: usize, const DIM: usize> {
    size: [usize; DIM],
    reference: &'a T,
}

impl<'a, T: Broadcastable, const DIM: usize> TensorDimension for LazyUpdim<'a, T, { T::DIM }, DIM> {
    const DIM: usize = DIM;
}

impl<'a, T: Broadcastable, const DIM: usize> TensorSize for LazyUpdim<'a, T, { T::DIM }, DIM> {
    fn size(&self) -> [usize; DIM] {
        //~^ ERROR: method not compatible with trait
        self.size
    }
}

impl<'a, T: Broadcastable, const DIM: usize> Broadcastable for LazyUpdim<'a, T, { T::DIM }, DIM> {
    type Element = T::Element;
    fn bget(&self, index: [usize; DIM]) -> Option<Self::Element> {
        //~^ ERROR: method not compatible with trait
        assert!(DIM >= T::DIM);
        if !self.inbounds(index) {
            //~^ ERROR: unconstrained generic constant
            //~| ERROR: mismatched types
            return None;
        }
        let size = self.size();
        //~^ ERROR: unconstrained generic constant
        let newindex: [usize; T::DIM] = Default::default();
        //~^ ERROR: the trait bound
        self.reference.bget(newindex)
    }
}

struct BMap<'a, R, T: Broadcastable, F: Fn(T::Element) -> R, const DIM: usize> {
    reference: &'a T,
    closure: F,
}

impl<'a, R, T: Broadcastable, F: Fn(T::Element) -> R, const DIM: usize> TensorDimension
    for BMap<'a, R, T, F, DIM>
{
    const DIM: usize = DIM;
}
impl<'a, R, T: Broadcastable, F: Fn(T::Element) -> R, const DIM: usize> TensorSize
    for BMap<'a, R, T, F, DIM>
{
    fn size(&self) -> [usize; DIM] {
        //~^ ERROR: method not compatible with trait
        self.reference.size()
        //~^ ERROR: unconstrained generic constant
        //~| ERROR: mismatched types
    }
}

impl<'a, R, T: Broadcastable, F: Fn(T::Element) -> R, const DIM: usize> Broadcastable
    for BMap<'a, R, T, F, DIM>
{
    type Element = R;
    fn bget(&self, index: [usize; DIM]) -> Option<Self::Element> {
        //~^ ERROR: method not compatible with trait
        self.reference.bget(index).map(&self.closure)
        //~^ ERROR: unconstrained generic constant
        //~| ERROR: mismatched types
    }
}

impl<T> TensorDimension for Vec<T> {
    const DIM: usize = 1;
}
impl<T> TensorSize for Vec<T> {
    fn size(&self) -> [usize; 1] {
        //~^ ERROR: method not compatible with trait
        [self.len()]
    }
}
impl<T: Clone> Broadcastable for Vec<T> {
    type Element = T;
    fn bget(&self, index: [usize; 1]) -> Option<T> {
        //~^ ERROR: method not compatible with trait
        self.get(index[0]).cloned()
    }
}

fn main() {
    let v = vec![1, 2, 3];
    let bv = v.lazy_updim([3, 4]);
    let bbv = bv.bmap(|x| x * x);

    println!("The size of v is {:?}", bbv.bget([0, 2]).expect("Out of bounds."));
}
