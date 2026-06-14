// Regression test for <https://github.com/rust-lang/rust/issues/120337>.
//
// This checks that const eval doesn't cause an ICE when reading an uninhabited
// variant.
#![feature(never_type)]

#[derive(Copy, Clone)]
pub enum E { A(!), }

pub union U { u: (), e: E, }

pub const C: () = {
    let E::A(ref a) = unsafe { &(&U { u: () }).e };
    //~^ ERROR: read discriminant of an uninhabited enum variant
};

fn main() {}
