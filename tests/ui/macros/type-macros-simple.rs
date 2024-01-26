//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_local_definitions)]

macro_rules! Tuple {
    { $A:ty,$B:ty } => { ($A, $B) }
}

fn main() {
    let x: Tuple!(i32, i32) = (1, 2);
}

fn issue_36540() {
    let i32 = 0;
    macro_rules! m { () => { i32 } }
    struct S<T = m!()>(m!(), T) where T: Trait<m!()>;

    let x: m!() = m!();
    std::cell::Cell::<m!()>::new(m!());
    impl<T> std::ops::Index<m!()> for dyn Trait<(m!(), T)>
        where T: Trait<m!()>
    {
        type Output = m!();
        fn index(&self, i: m!()) -> &m!() {
            unimplemented!()
        }
    }
}

trait Trait<T> {}
impl Trait<i32> for i32 {}
