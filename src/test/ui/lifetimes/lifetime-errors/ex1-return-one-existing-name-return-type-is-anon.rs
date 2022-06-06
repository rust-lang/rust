// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Foo {
  field: i32
}

impl Foo {
  fn foo<'a>(&self, x: &'a i32) -> &i32 {

    x
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ ERROR lifetime may not live long enough

  }

}

fn main() { }
