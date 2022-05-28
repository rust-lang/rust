// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo(x:Box<dyn Fn(&u8, &u8)> , y: Vec<&u8>, z: &u8) {
  y.push(z);
  //[base]~^ ERROR lifetime mismatch
  //[nll]~^^ ERROR lifetime may not live long enough
  //[nll]~| ERROR cannot borrow
}

fn main() { }
