#![feature(raw_ref_op)]

fn main() {
  let a1 = [4,5,6];
  let p1 = &raw const a1;
  let _ = unsafe { p1.get_unchecked(1) };

  let mut a2 = [4,5,6];
  let p2 = &raw mut a2;
  let _ = unsafe { p2.get_unchecked(1) };

  let mut a3 = [4,5,6];
  let p3 = &raw mut a3;
  let _ = unsafe { p3.get_unchecked_mut(1) };

  let a4 = [4,5,6];
  let p4 = &raw const a4;
  let _ = unsafe { p4.get_unchecked_mut(1) };
  //~^ ERROR cannot borrow
}
