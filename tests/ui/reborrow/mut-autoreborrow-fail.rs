// Known bug: ideally, this would compile.

fn assert_fnmut(_: &mut impl FnMut()) {}

fn main() {
    let x = &mut Box::new(0);
    let mut f = || {
        //~^ ERROR expected a closure that implements the `FnMut` trait, but this closure only implements `FnOnce` [E0525]
        let _y = x;
    };
    f();
    f();
    assert_fnmut(&mut f);
}
