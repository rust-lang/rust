// check-pass

// Some trait with a function that returns a slice:
pub trait AsSlice {
    type Element;
    fn as_slice(&self) -> &[Self::Element];
}

// Some type
pub struct A<Cont>(Cont);

// Here we say that if A wraps a slice, then it implements AsSlice
impl<'a, Element> AsSlice for A<&'a [Element]> {
    type Element = Element;
    fn as_slice(&self) -> &[Self::Element] {
        self.0
    }
}

impl<Cont> A<Cont> {
    // We want this function to work
    pub fn failing<Coef>(&self)
    where
        Self: AsSlice<Element = Coef>,
    {
        self.as_ref_a().as_ref_a();
    }

    pub fn as_ref_a<Coef>(&self) -> A<&[<Self as AsSlice>::Element]>
    where
        Self: AsSlice<Element = Coef>,
    {
        A(self.as_slice())
    }
}

fn main() {}
