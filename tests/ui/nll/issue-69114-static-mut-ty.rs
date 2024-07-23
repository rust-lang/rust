// Check that borrowck ensures that `static mut` items have the expected type.

static FOO: u8 = 42;
static mut BAR: &'static u8 = &FOO;
static mut BAR_ELIDED: &u8 = &FOO;

fn main() {
    unsafe {
        println!("{} {}", BAR, BAR_ELIDED);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        //~^^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        set_bar();
        set_bar_elided();
        println!("{} {}", BAR, BAR_ELIDED);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        //~^^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
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
