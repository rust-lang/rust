// Test that we fail to promote the constant here which has a `ref
// mut` borrow.

fn gimme_static_mut_let() -> &'static mut u32 {
    let ref mut x = 1234543;
    x //~ ERROR
}

fn gimme_static_mut_let_nested() -> &'static mut u32 {
    let (ref mut x, ) = (1234543, );
    x //~ ERROR
}

fn gimme_static_mut_match() -> &'static mut u32 {
    match 1234543 { //~ ERROR
        ref mut x => x
    }
}

fn gimme_static_mut_match_nested() -> &'static mut u32 {
    match (123443,) { //~ ERROR
        (ref mut x,) => x,
    }
}

fn gimme_static_mut_ampersand() -> &'static mut u32 {
    &mut 1234543 //~ ERROR
}

fn main() {
}
