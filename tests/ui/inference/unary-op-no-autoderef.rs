use std::ops::{Deref, Index, Neg, Not};

struct Inner {
    value: u32,
}

// `Inner` implements everything that the operators and place projections look
// up, so the contrast below is only about whether `Wrapper` is autoderef'd.
impl Not for Inner {
    type Output = Inner;

    fn not(self) -> Inner {
        self
    }
}

impl Neg for Inner {
    type Output = Inner;

    fn neg(self) -> Inner {
        self
    }
}

impl Index<usize> for Inner {
    type Output = u32;

    fn index(&self, _: usize) -> &u32 {
        &self.value
    }
}

struct Wrapper(Inner);

impl Deref for Wrapper {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        &self.0
    }
}

fn main() {
    let wrapper = Wrapper(Inner { value: 0 });

    // Field access and indexing are place projections, so they autoderef
    // `Wrapper` to `Inner` and both of these compile.
    let _ = wrapper.value;
    let _ = wrapper[0];

    // `!` and `-` are value operators instead: they look up `Not` and `Neg` on
    // the operand type itself and do not autoderef it. `Wrapper` implements
    // neither trait, so both are rejected even though `Inner` does implement
    // them and `Wrapper` derefs to `Inner`.
    let _ = !wrapper;
    //~^ ERROR cannot apply unary operator `!` to type `Wrapper`

    let _ = -wrapper;
    //~^ ERROR cannot apply unary operator `-` to type `Wrapper`
}
