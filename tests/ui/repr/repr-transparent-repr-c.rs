#[repr(C)]
pub struct ReprC1Zst {
    pub _f: (),
}

pub struct ReprRust<T>(T);

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

#[repr(transparent)]
pub struct T8(Sized, ReprRust<ReprC1Zst>);
#[repr(transparent)]
pub struct T9(Sized, (ReprC1Zst,));
#[repr(transparent)]
pub struct T10(Sized, [u8; 0]);
//~^ WARN needs at most one non-trivial field
//~| WARN this was previously accepted by the compiler
#[repr(transparent)]
pub struct T11(Sized, !);

fn main() {}
