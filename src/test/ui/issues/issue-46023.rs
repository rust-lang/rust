// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let x = 0;

    (move || {
        x = 1;
        //[mir]~^ ERROR cannot assign to `x`, as it is not declared as mutable [E0594]
        //[ast]~^^ ERROR cannot assign to captured outer variable in an `FnMut` closure [E0594]
    })()
}
