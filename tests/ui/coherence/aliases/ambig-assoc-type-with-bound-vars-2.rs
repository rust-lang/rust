// Another regression test for #114061.

pub trait WhereBound {}
impl WhereBound for () {}

pub trait WithAssoc<'a> {
    type Assoc;
}

// The two impls of `Trait` overlap
pub trait Trait {}
impl<T> Trait for T
where
    T: 'static,
    for<'a> T: WithAssoc<'a>,
    // This bound was previously treated as knowable
    for<'a> Box<<T as WithAssoc<'a>>::Assoc>: WhereBound,
{
}
impl<T> Trait for Box<T> {}
//~^ ERROR conflicting implementations of trait `Trait`

fn main() {}
