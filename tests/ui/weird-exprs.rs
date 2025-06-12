//@ run-pass

#![feature(coroutines)]

#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(redundant_semicolons)]
#![allow(unreachable_code)]
#![allow(unused_braces, unused_must_use, unused_parens)]
#![allow(uncommon_codepoints, confusable_idents)]
#![allow(unused_imports)]
#![allow(unreachable_patterns)]

#![recursion_limit = "256"]

extern crate core;
use std::cell::Cell;
use std::mem::swap;
use std::ops::Deref;

// Just a grab bag of stuff that you wouldn't want to actually write.

fn strange() -> bool { let _x: bool = return true; }

fn funny() {
    fn f(_x: ()) { }
    f(return);
}

fn what() {
    fn the(x: &Cell<bool>) {
        return while !x.get() { x.set(true); };
    }
    let i = &Cell::new(false);
    let dont = {||the(i)};
    dont();
    assert!(i.get());
}

fn zombiejesus() {
    loop {
        while (return) {
            if (return) {
                match (return) {
                    1 => {
                        if (return) {
                            return
                        } else {
                            return
                        }
                    }
                    _ => { return }
                };
            } else if (return) {
                return;
            }
        }
        if (return) { break; }
    }
}

fn notsure() {
    let mut _x: isize;
    let mut _y = (_x = 0) == (_x = 0);
    let mut _z = (_x = 0) < (_x = 0);
    let _a = (_x += 0) == (_x = 0);
    let _b = swap(&mut _y, &mut _z) == swap(&mut _y, &mut _z);
}

fn canttouchthis() -> usize {
    fn p() -> bool { true }
    let _a = (assert!(true) == (assert!(p())));
    let _c = (assert!(p()) == ());
    let _b: bool = (println!("{}", 0) == (return 0));
}

fn angrydome() {
    loop { if break { } }
    let mut i = 0;
    loop { i += 1; if i == 1 { match (continue) { 1 => { }, _ => panic!("wat") } }
      break; }
}

fn evil_lincoln() { let _evil: () = println!("lincoln"); }

fn dots() {
    assert_eq!(String::from(".................................................."),
               format!("{:?}", .. .. .. .. .. .. .. .. .. .. .. .. ..
                               .. .. .. .. .. .. .. .. .. .. .. ..));
}

fn u8(u8: u8) {
    if u8 != 0u8 {
        assert_eq!(8u8, {
            macro_rules! u8 {
                (u8) => {
                    mod u8 {
                        pub fn u8<'u8: 'u8 + 'u8>(u8: &'u8 u8) -> &'u8 u8 {
                            "u8";
                            u8
                        }
                    }
                };
            }

            u8!(u8);
            let &u8: &u8 = u8::u8(&8u8);
            crate::u8(0u8);
            u8
        });
    }
}

fn fishy() {
    assert_eq!(String::from("><>"),
               String::<>::from::<>("><>").chars::<>().rev::<>().collect::<String>());
}

fn union() {
    union union<'union> { union: &'union union<'union>, }
}

fn special_characters() {
    let val = !((|(..):(_,_),(|__@_|__)|__)((&*"\\",'🤔')/**/,{})=={&[..=..][..];})//
    ;
    assert!(!val);
}

fn punch_card() -> impl std::fmt::Debug {
    ..=..=.. ..    .. .. .. ..    .. .. .. ..    .. .. .. ..
    ..=.. ..=..    .. .. .. ..    .. .. .. ..    .. ..=.. ..
    ..=.. ..=..    ..=.. ..=..    .. ..=..=..    ..=..=..=..
    ..=..=.. ..    ..=.. ..=..    ..=.. .. ..    .. ..=.. ..
    ..=.. ..=..    ..=.. ..=..    .. ..=.. ..    .. ..=.. ..
    ..=.. ..=..    ..=.. ..=..    .. .. ..=..    .. ..=.. ..
    ..=.. ..=..    .. ..=..=..    ..=..=.. ..    .. ..=..=..
}

fn r#match() {
    let val: () = match match match match match () {
        () => ()
    } {
        () => ()
    } {
        () => ()
    } {
        () => ()
    } {
        () => ()
    };
    assert_eq!(val, ());
}

fn i_yield() {
    #[coroutine]
    static || {
        yield yield yield yield yield yield yield yield yield;
    };
}

fn match_nested_if() {
    let val = match () {
        () if if if if true {true} else {false} {true} else {false} {true} else {false} => true,
        _ => false,
    };
    assert!(val);
}

fn monkey_barrel() {
    let val: () = ()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=()=();
    assert_eq!(val, ());
}

fn 𝚌𝚘𝚗𝚝𝚒𝚗𝚞𝚎() {
    type 𝚕𝚘𝚘𝚙 = i32;
    fn 𝚋𝚛𝚎𝚊𝚔() -> 𝚕𝚘𝚘𝚙 {
        let 𝚛𝚎𝚝𝚞𝚛𝚗 = 42;
        return 𝚛𝚎𝚝𝚞𝚛𝚗;
    }
    assert_eq!(loop {
        break 𝚋𝚛𝚎𝚊𝚔 ();
    }, 42);
}

fn function() {
    struct foo;
    impl Deref for foo {
        type Target = fn() -> Self;
        fn deref(&self) -> &Self::Target {
            &((|| foo) as _)
        }
    }
    let foo = foo () ()() ()()() ()()()() ()()()()();
}

fn bathroom_stall() {
    let mut i = 1;
    matches!(2, _|_|_|_|_|_ if (i+=1) != (i+=1));
    assert_eq!(i, 13);
}

fn closure_matching() {
    let x = |_| Some(1);
    let (|x| x) = match x(..) {
        |_| Some(2) => |_| Some(3),
        |_| _ => unreachable!(),
    };
    assert!(matches!(x(..), |_| Some(4)));
}

fn semisemisemisemisemi() {
    ;;;;;;; ;;;;;;; ;;;    ;;; ;;
    ;;      ;;      ;;;;  ;;;; ;;
    ;;;;;;; ;;;;;   ;; ;;;; ;; ;;
         ;; ;;      ;;  ;;  ;; ;;
    ;;;;;;; ;;;;;;; ;;      ;; ;;
}

fn useful_syntax() {
    use {{std::{{collections::{{HashMap}}}}}};
    use ::{{{{core}, {std}}}};
    use {{::{{core as core2}}}};
}

fn infcx() {
    pub mod cx {
        pub mod cx {
            pub use super::cx;
            pub struct Cx;
        }
    }
    let _cx: cx::cx::Cx = cx::cx::cx::cx::cx::Cx;
}

fn return_already() -> impl std::fmt::Debug {
    loop {
        return !!!!!!!
        break !!!!!!1111
    }
}

fn fake_macros() -> impl std::fmt::Debug {
    loop {
        if! {
            match! (
                break! {
                    return! {
                        1337
                    }
                }
            )

            {}
        }

        {}
    }
}

fn fish_fight() {
    trait Rope {
        fn _____________<U>(_: Self, _: U) where Self: Sized {}
    }

    struct T;

    impl Rope for T {}

    fn tug_o_war(_: impl Fn(T, T)) {}

    tug_o_war(<T>::_____________::<T>);
}

pub fn main() {
    strange();
    funny();
    what();
    zombiejesus();
    notsure();
    canttouchthis();
    angrydome();
    evil_lincoln();
    dots();
    u8(8u8);
    fishy();
    union();
    special_characters();
    punch_card();
    r#match();
    i_yield();
    match_nested_if();
    monkey_barrel();
    𝚌𝚘𝚗𝚝𝚒𝚗𝚞𝚎();
    function();
    bathroom_stall();
    closure_matching();
    semisemisemisemisemi();
    useful_syntax();
    infcx();
    return_already();
    fake_macros();
    fish_fight();
}
