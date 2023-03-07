// error-pattern: evaluation of constant value failed

const unsafe fn foo(x: bool) -> bool {
    match x {
        true => true,
        false => std::hint::unreachable_unchecked(),
    }
}

const BAR: bool = unsafe { foo(false) };

fn main() {
    assert_eq!(BAR, true);
}
