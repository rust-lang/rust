// Check that borrowck ensures that `static mut` items have the expected type.

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static FOO: u8 = 42;
static mut BAR: &'static u8 = &FOO;
static mut BAR_ELIDED: &u8 = &FOO;

fn main() {
    unsafe {
        println!("{} {}", BAR, BAR_ELIDED);
        set_bar();
        set_bar_elided();
        println!("{} {}", BAR, BAR_ELIDED);
    }
}

fn set_bar() {
    let n = 42;
    unsafe {
        BAR = &n;
        //~^ ERROR does not live long enough
    }
}

fn set_bar_elided() {
    let n = 42;
    unsafe {
        BAR_ELIDED = &n;
        //~^ ERROR does not live long enough
    }
}
