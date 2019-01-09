// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

static NUM: i32 = 18;

fn main() {
    NUM = 20; //[ast]~ ERROR E0594
              //[mir]~^ ERROR cannot assign to immutable static item `NUM`
}
