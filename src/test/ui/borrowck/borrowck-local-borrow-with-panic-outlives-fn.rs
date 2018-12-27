// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn cplusplus_mode_exceptionally_unsafe(x: &mut Option<&'static mut isize>) {
    let mut z = (0, 0);
    *x = Some(&mut z.1);
    //[ast]~^ ERROR `z.1` does not live long enough [E0597]
    //[mir]~^^ ERROR `z.1` does not live long enough [E0597]
    panic!("catch me for a dangling pointer!")
}

fn main() {
    cplusplus_mode_exceptionally_unsafe(&mut None);
}
