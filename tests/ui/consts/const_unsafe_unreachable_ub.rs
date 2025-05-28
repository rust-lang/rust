const unsafe fn foo(x: bool) -> bool {
    match x {
        true => true,
        false => std::hint::unreachable_unchecked(),
        //~^ NOTE inside `foo`
        //~| NOTE the failure occurred here
    }
}

const BAR: bool = unsafe { foo(false) };
//~^ NOTE evaluation of constant value failed
//~| ERROR entering unreachable code

fn main() {
    assert_eq!(BAR, true);
}
