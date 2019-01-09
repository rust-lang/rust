// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let mut _a = 3;
    let b = &mut _a;
    {
        let c = &*b;
        _a = 4; //[ast]~ ERROR cannot assign to `_a`
        //[mir]~^ ERROR cannot assign to `_a` because it is borrowed
        drop(c);
    }
    drop(b);
}
