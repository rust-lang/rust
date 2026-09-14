#![warn(clippy::mem_replace_option_with_none)]

use std::mem;

fn main() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    //~^ mem_replace_option_with_none
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
    //~^ mem_replace_option_with_none
}

fn issue9824() {
    struct Foo<'a>(Option<&'a str>);
    impl<'a> std::ops::Deref for Foo<'a> {
        type Target = Option<&'a str>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<'a> std::ops::DerefMut for Foo<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    struct Bar {
        opt: Option<u8>,
    }

    let mut f = Foo(Some("foo"));
    let mut b = Bar { opt: Some(1) };

    let _ = std::mem::replace(&mut f.0, None);
    //~^ mem_replace_option_with_none
    let _ = std::mem::replace(&mut *f, None);
    //~^ mem_replace_option_with_none
    let _ = std::mem::replace(&mut b.opt, None);
    //~^ mem_replace_option_with_none
}
