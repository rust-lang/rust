//@ edition: 2021

// Nested items/closures should be treated as "holes", so that their spans are
// not displayed as executable code in the enclosing function.

use core::hint::black_box;

fn main() {
    black_box(());

    // Splitting this across multiple lines makes it easier to see where the
    // coverage mapping regions begin and end.
    #[rustfmt::skip]
    let _closure =
        |
            _arg: (),
        |
        {
            black_box(());
        }
        ;

    black_box(());

    fn _unused_fn() {}

    black_box(());

    struct MyStruct {
        _x: u32,
        _y: u32,
    }

    black_box(());

    impl MyStruct {
        fn _method(&self) {}
    }

    black_box(());

    macro_rules! _my_macro {
        () => {};
    }

    black_box(());

    #[rustfmt::skip]
    let _const =
        const
        {
            7 + 4
        }
        ;

    black_box(());

    #[rustfmt::skip]
    let _async =
        async
        {
            7 + 4
        }
        ;

    black_box(());
}
