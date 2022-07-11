pub unsafe fn test() {
    let pointer = 1u32 as *const _;
    //~^ ERROR cannot cast to a pointer of an unknown kind
    core::arch::asm!(
        "nop",
        in("eax") pointer,
    );
}

fn main() {}
