#![feature(never_type)]
#[derive(Copy, Clone)]
pub enum E { A(!), }
pub union U { u: (), e: E, }
pub const C: () = { //~ ERROR evaluation of constant value failed
    let E::A(ref a) = unsafe { &(&U { u: () }).e};
};

fn main() {}
