// Make sure that an invalid inherent impl doesn't totally clobber all of the
// other inherent impls, which lead to mysterious method/assoc-item probing errors.

impl () {}
//~^ ERROR cannot define inherent `impl` for primitive types

struct W;
impl W {
    const CONST: u32 = 0;
}

fn main() {
    let _ = W::CONST;
}
