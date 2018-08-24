// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn cplusplus_mode(x: isize) -> &'static isize {
    &x
    //[ast]~^ ERROR `x` does not live long enough [E0597]
    //[mir]~^^ ERROR `x` does not live long enough [E0597]
}

fn main() {}
