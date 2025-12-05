#![allow(unused, clippy::needless_lifetimes)]
#![warn(
    clippy::style,
    clippy::mem_replace_option_with_none,
    clippy::mem_replace_with_default
)]
#![feature(lang_items)]
#![no_std]

use core::mem;
use core::panic::PanicInfo;

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

fn replace_option_with_none() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    //~^ mem_replace_option_with_none
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
    //~^ mem_replace_option_with_none
}

fn replace_with_default() {
    let mut refstr = "hello";
    let _ = mem::replace(&mut refstr, "");
    //~^ mem_replace_with_default

    let mut slice: &[i32] = &[1, 2, 3];
    let _ = mem::replace(&mut slice, &[]);
    //~^ mem_replace_with_default
}

// lint is disabled for primitives because in this case `take`
// has no clear benefit over `replace` and sometimes is harder to read
fn dont_lint_primitive() {
    let mut pbool = true;
    let _ = mem::replace(&mut pbool, false);

    let mut pint = 5;
    let _ = mem::replace(&mut pint, 0);
}

fn main() {
    replace_option_with_none();
    replace_with_default();
    dont_lint_primitive();
}

fn issue9824() {
    struct Foo<'a>(Option<&'a str>);
    impl<'a> core::ops::Deref for Foo<'a> {
        type Target = Option<&'a str>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<'a> core::ops::DerefMut for Foo<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    struct Bar {
        opt: Option<u8>,
        val: u8,
    }

    let mut f = Foo(Some("foo"));
    let mut b = Bar { opt: Some(1), val: 12 };

    // replace option with none
    let _ = mem::replace(&mut f.0, None);
    //~^ mem_replace_option_with_none
    let _ = mem::replace(&mut *f, None);
    //~^ mem_replace_option_with_none
    let _ = mem::replace(&mut b.opt, None);
    //~^ mem_replace_option_with_none
    // replace with default
    let _ = mem::replace(&mut b.val, u8::default());
}
