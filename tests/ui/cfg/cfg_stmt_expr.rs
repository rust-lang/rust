//@ run-pass
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![deny(non_snake_case)]
#![feature(stmt_expr_attributes)]

fn main() {
    let a = 413;
    #[cfg(false)]
    let a = ();
    assert_eq!(a, 413);

    let mut b = 612;
    #[cfg(false)]
    {
        b = 1111;
    }
    assert_eq!(b, 612);

    #[cfg(false)]
    undefined_fn();

    #[cfg(false)]
    undefined_macro!();
    #[cfg(false)]
    undefined_macro![];
    #[cfg(false)]
    undefined_macro!{};

    // pretty printer bug...
    // #[cfg(false)]
    // undefined_macro!{}

    let () = (#[cfg(false)] 341,); // Should this also work on parens?
    let t = (1, #[cfg(false)] 3, 4);
    assert_eq!(t, (1, 4));

    let f = |_: u32, _: u32| ();
    f(2, 1, #[cfg(false)] 6);

    let _: u32 = a.clone(#[cfg(false)] undefined);

    let _: [(); 0] = [#[cfg(false)] 126];
    let t = [#[cfg(false)] 1, 2, 6];
    assert_eq!(t, [2, 6]);

    {
        let r;
        #[cfg(false)]
        (r = 5);
        #[cfg(not(FALSE))]
        (r = 10);
        assert_eq!(r, 10);
    }

    // check that macro expanded code works

    macro_rules! if_cfg {
        ($cfg:meta? $ib:block else $eb:block) => {
            {
                let r;
                #[cfg($cfg)]
                (r = $ib);
                #[cfg(not($cfg))]
                (r = $eb);
                r
            }
        }
    }

    let n = if_cfg!(FALSE? {
        413
    } else {
        612
    });

    assert_eq!((#[cfg(false)] 1, #[cfg(not(FALSE))] 2), (2,));
    assert_eq!(n, 612);

    // check that lints work

    #[allow(non_snake_case)]
    let FOOBAR: () = {
        fn SYLADEX() {}
    };

    #[allow(non_snake_case)]
    {
        fn CRUXTRUDER() {}
    }
}
