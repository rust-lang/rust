// Regression test for #117151, this used to hang the compiler

pub type ISO<A: 'static, B: 'static> = (Box<dyn Fn(A) -> B>, Box<dyn Fn(B) -> A>);
pub fn iso<A: 'static, B: 'static, F1, F2>(a: F1, b: F2) -> ISO<A, B>
where
    F1: 'static + Fn(A) -> B,
    F2: 'static + Fn(B) -> A,
{
    (Box::new(a), Box::new(b))
}
pub fn iso_un_option<A: 'static, B: 'static>(i: ISO<Option<A>, Option<B>>) -> ISO<A, B> {
    let (ab, ba) = (i.ab, i.ba);
    //~^ ERROR no field `ab` on type
    //~| ERROR no field `ba` on type
    let left = move |o_a| match o_a {
        //~^ ERROR overflow assigning `_` to `Option<_>`
        None => panic!("absurd"),
        Some(a) => a,
    };
    let right = move |o_b| match o_b {
        None => panic!("absurd"),
        Some(b) => b,
    };
    iso(left, right)
}

fn main() {}
