// This test should never pass!

#![feature(type_alias_impl_trait)]

pub trait Captures<'a> {}
impl<T> Captures<'_> for T {}

pub struct MyTy<'a, 'b>(Option<*mut &'a &'b ()>);
unsafe impl Send for MyTy<'_, 'static> {}

pub type Step1<'a, 'b: 'a> = impl Sized + Captures<'b> + 'a;
#[define_opaque(Step1)]
pub fn step1<'a, 'b: 'a>() -> Step1<'a, 'b> {
    MyTy::<'a, 'b>(None)
}

pub type Step2<'a> = impl Send + 'a;

// Although `Step2` is WF at the definition site, it's not WF in its
// declaration site (above). We check this in `check_opaque_meets_bounds`,
// which must remain sound.
#[define_opaque(Step2)]
pub fn step2<'a, 'b: 'a>() -> Step2<'a>
where
    Step1<'a, 'b>: Send,
{
    step1::<'a, 'b>()
    //~^ ERROR hidden type for `Step2<'a>` captures lifetime that does not appear in bounds
}

fn step3<'a, 'b>() {
    fn is_send<T: Send>() {}
    is_send::<Step2<'a>>();
}

fn main() {}
