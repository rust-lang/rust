pub fn iso<A, B, F1, F2>(a: F1, b: F2) -> (Box<dyn Fn(A) -> B>, Box<dyn Fn(B) -> A>)
    where
        F1: (Fn(A) -> B) + 'static,
        F2: (Fn(B) -> A) + 'static,
{
    (Box::new(a), Box::new(b))
}
pub fn iso_un_option<A, B>() -> (Box<dyn Fn(A) -> B>, Box<dyn Fn(B) -> A>) {
    let left = |o_a: Option<_>| o_a.unwrap();
    //~^ ERROR overflow
    let right = |o_b: Option<_>| o_b.unwrap();
    iso(left, right)
}

fn main() {}
