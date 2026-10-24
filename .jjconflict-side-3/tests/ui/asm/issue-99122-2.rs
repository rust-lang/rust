//@ check-pass
//@ needs-asm-support
//@ only-x86_64

// This demonstrates why we need to erase regions before sized check in intrinsicck

struct NoCopy;

struct Wrap<'a, T, Tail: ?Sized>(&'a T, Tail);

pub unsafe fn test() {
    let i = NoCopy;
    let j = Wrap(&i, ());
    let pointer = &j as *const _;
    core::arch::asm!(
        "nop",
        in("eax") pointer,
    );
}

fn main() {}
