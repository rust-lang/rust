// revisions: x86_64_mirunsafeck aarch64_mirunsafeck x86_64_thirunsafeck aarch64_thirunsafeck

// [x86_64_thirunsafeck] compile-flags: -Z thir-unsafeck --target x86_64-unknown-linux-gnu
// [aarch64_thirunsafeck] compile-flags: -Z thir-unsafeck --target aarch64-unknown-linux-gnu
// [x86_64_mirunsafeck] compile-flags: --target x86_64-unknown-linux-gnu
// [aarch64_mirunsafeck] compile-flags: --target aarch64-unknown-linux-gnu

// [x86_64_thirunsafeck] needs-llvm-components: x86
// [x86_64_mirunsafeck] needs-llvm-components: x86
// [aarch64_thirunsafeck] needs-llvm-components: aarch64
// [aarch64_mirunsafeck] needs-llvm-components: aarch64

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! global_asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}

fn main() {
    let mut foo = 0;
    unsafe {
        asm!("{}");
        //~^ ERROR invalid reference to argument at index 0
        asm!("{1}", in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ ERROR argument never used
        asm!("{a}");
        //~^ ERROR there is no argument named `a`
        asm!("{}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 0
        //~^^ ERROR argument never used
        asm!("{1}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ ERROR named argument never used
        #[cfg(any(x86_64_thirunsafeck, x86_64_mirunsafeck))]
        asm!("{}", in("eax") foo);
        //[x86_64_thirunsafeck,x86_64_mirunsafeck]~^ ERROR invalid reference to argument at index 0
        #[cfg(any(aarch64_thirunsafeck, aarch64_mirunsafeck))]
        asm!("{}", in("x0") foo);
        //[aarch64_thirunsafeck,aarch64_mirunsafeck]~^ ERROR invalid reference to argument at index 0
        asm!("{:foo}", in(reg) foo);
        //~^ ERROR asm template modifier must be a single character
        asm!("", in(reg) 0, in(reg) 1);
        //~^ ERROR multiple unused asm arguments
    }
}

const FOO: i32 = 1;
global_asm!("{}");
//~^ ERROR invalid reference to argument at index 0
global_asm!("{1}", const FOO);
//~^ ERROR invalid reference to argument at index 1
//~^^ ERROR argument never used
global_asm!("{a}");
//~^ ERROR there is no argument named `a`
global_asm!("{}", a = const FOO);
//~^ ERROR invalid reference to argument at index 0
//~^^ ERROR argument never used
global_asm!("{1}", a = const FOO);
//~^ ERROR invalid reference to argument at index 1
//~^^ ERROR named argument never used
global_asm!("{:foo}", const FOO);
//~^ ERROR asm template modifier must be a single character
global_asm!("", const FOO, const FOO);
//~^ ERROR multiple unused asm arguments
