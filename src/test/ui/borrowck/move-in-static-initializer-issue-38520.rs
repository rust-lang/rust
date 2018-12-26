// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

// Regression test for #38520. Check that moves of `Foo` are not
// permitted as `Foo` is not copy (even in a static/const
// initializer).

struct Foo(usize);

const fn get(x: Foo) -> usize {
    x.0
}

const X: Foo = Foo(22);
static Y: usize = get(*&X); //[ast]~ ERROR E0507
                            //[mir]~^ ERROR [E0507]
const Z: usize = get(*&X); //[ast]~ ERROR E0507
                           //[mir]~^ ERROR [E0507]

fn main() {
}
