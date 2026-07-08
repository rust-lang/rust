#[repr(C)]
pub struct ReprC1Zst {
    pub _f: (),
}

pub type Sized = i32;

#[repr(transparent)]
pub struct T1(ReprC1Zst);
#[repr(transparent)]
pub struct T2((), ReprC1Zst);
#[repr(transparent)]
pub struct T3(ReprC1Zst, ());

#[repr(transparent)]
pub struct T5(Sized, ReprC1Zst);
//~^ ERROR needs at most one non-trivial field

#[repr(transparent)]
pub struct T6(ReprC1Zst, Sized);
//~^ ERROR needs at most one non-trivial field

#[repr(transparent)]
pub struct T7(T1, [Sized; 0]); // still wrong, even when the repr(C) is hidden inside another type
//~^ ERROR needs at most one non-trivial field

fn main() {}
