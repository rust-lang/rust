//@ needs-asm-support
//@ build-pass

use std::arch::asm;

fn foofoo<const N: usize>() {}

unsafe fn foo<const N: usize>() {
    asm!("/* {0} */", const N);
    asm!("/* {0} */", const N + 1);
    asm!("/* {0} */", sym foofoo::<N>);
}

fn barbar<T>() {}

unsafe fn bar<T>() {
    asm!("/* {0} */", const std::mem::size_of::<T>());
    asm!("/* {0} */", const std::mem::size_of::<(T, T)>());
    asm!("/* {0} */", sym barbar::<T>);
    asm!("/* {0} */", sym barbar::<(T, T)>);
}

fn main() {
    unsafe {
        foo::<0>();
        bar::<usize>();
    }
}
