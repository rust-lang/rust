// run-pass
// compile-flags:-Zmir-opt-level=2

pub enum Enum {
    A,
    B,
}

trait SliceIndex {
    type Output;
    fn get(&self) -> &Self::Output;
}

impl SliceIndex for usize {
    type Output = Enum;
    #[inline(never)]
    fn get(&self) -> &Enum {
        &Enum::A
    }
}

#[inline(always)]
fn index<T: SliceIndex>(t: &T) -> &T::Output {
    t.get()
}

fn main() {
    match *index(&0) { Enum::A => true, _ => false };
}
