//@ needs-asm-support
//@ only-x86_64

struct W<T: ?Sized>(Oops);
//~^ ERROR cannot find type `Oops` in this scope

unsafe fn test() {
    let j = W(());
    let pointer = &j as *const _;
    core::arch::asm!(
        "nop",
        in("eax") pointer,
    );
}

fn main() {}
