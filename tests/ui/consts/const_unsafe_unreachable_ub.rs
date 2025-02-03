const unsafe fn foo(x: bool) -> bool {
    match x {
        true => true,
        false => std::hint::unreachable_unchecked(), //~ NOTE inside `foo`
    }
}

const BAR: bool = unsafe { foo(false) };
//~^ ERROR evaluation of constant value failed
//~| NOTE entering unreachable code
//~| NOTE inside `unreachable_unchecked`

fn main() {
    assert_eq!(BAR, true);
}
