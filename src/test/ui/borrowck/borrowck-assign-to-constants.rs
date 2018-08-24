// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

static foo: isize = 5;

fn main() {
    // assigning to various global constants
    foo = 6; //[ast]~ ERROR cannot assign to immutable static item
             //[mir]~^ ERROR cannot assign to immutable static item `foo`
}
