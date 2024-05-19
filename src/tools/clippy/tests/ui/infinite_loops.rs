//@no-rustfix
//@aux-build:proc_macros.rs

#![allow(clippy::never_loop)]
#![warn(clippy::infinite_loop)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn do_something() {}

fn no_break() {
    loop {
        //~^ ERROR: infinite loop detected
        do_something();
    }
}

fn all_inf() {
    loop {
        //~^ ERROR: infinite loop detected
        loop {
            //~^ ERROR: infinite loop detected
            loop {
                //~^ ERROR: infinite loop detected
                do_something();
            }
        }
        do_something();
    }
}

fn no_break_return_some_ty() -> Option<u8> {
    loop {
        do_something();
        return None;
    }
    loop {
        //~^ ERROR: infinite loop detected
        do_something();
    }
}

fn no_break_never_ret() -> ! {
    loop {
        do_something();
    }
}

fn no_break_never_ret_noise() {
    loop {
        fn inner_fn() -> ! {
            std::process::exit(0);
        }
        do_something();
    }
}

fn has_direct_break_1() {
    loop {
        do_something();
        break;
    }
}

fn has_direct_break_2() {
    'outer: loop {
        do_something();
        break 'outer;
    }
}

fn has_indirect_break_1(cond: bool) {
    'outer: loop {
        loop {
            if cond {
                break 'outer;
            }
        }
    }
}

fn has_indirect_break_2(stop_num: i32) {
    'outer: loop {
        for x in 0..5 {
            if x == stop_num {
                break 'outer;
            }
        }
    }
}

fn break_inner_but_not_outer_1(cond: bool) {
    loop {
        //~^ ERROR: infinite loop detected
        loop {
            if cond {
                break;
            }
        }
    }
}

fn break_inner_but_not_outer_2(cond: bool) {
    loop {
        //~^ ERROR: infinite loop detected
        'inner: loop {
            loop {
                if cond {
                    break 'inner;
                }
            }
        }
    }
}

fn break_outer_but_not_inner() {
    loop {
        loop {
            //~^ ERROR: infinite loop detected
            do_something();
        }
        break;
    }
}

fn can_break_both_inner_and_outer(cond: bool) {
    'outer: loop {
        loop {
            if cond {
                break 'outer;
            } else {
                break;
            }
        }
    }
}

fn break_wrong_loop(cond: bool) {
    // 'inner has statement to break 'outer loop, but it was breaked early by a labeled child loop
    'outer: loop {
        loop {
            //~^ ERROR: infinite loop detected
            'inner: loop {
                loop {
                    loop {
                        break 'inner;
                    }
                    break 'outer;
                }
            }
        }
    }
}

fn has_direct_return(cond: bool) {
    loop {
        if cond {
            return;
        }
    }
}

fn ret_in_inner(cond: bool) {
    loop {
        loop {
            if cond {
                return;
            }
        }
    }
}

enum Foo {
    A,
    B,
    C,
}

fn match_like() {
    let opt: Option<u8> = Some(1);
    loop {
        //~^ ERROR: infinite loop detected
        match opt {
            Some(v) => {
                println!("{v}");
            },
            None => {
                do_something();
            },
        }
    }

    loop {
        match opt {
            Some(v) => {
                println!("{v}");
            },
            None => {
                do_something();
                break;
            },
        }
    }

    let result: Result<u8, u16> = Ok(1);
    loop {
        let _val = match result {
            Ok(1) => 1 + 1,
            Ok(v) => v / 2,
            Err(_) => return,
        };
    }

    loop {
        let Ok(_val) = result else { return };
    }

    loop {
        let Ok(_val) = result.map(|v| 10) else { break };
    }

    loop {
        //~^ ERROR: infinite loop detected
        let _x = matches!(result, Ok(v) if v != 0).then_some(0);
    }

    loop {
        //~^ ERROR: infinite loop detected
        // This `return` does not return the function, so it doesn't count
        let _x = matches!(result, Ok(v) if v != 0).then(|| {
            if true {
                return;
            }
            do_something();
        });
    }

    let mut val = 0;
    let mut fooc = Foo::C;

    loop {
        val = match fooc {
            Foo::A => 0,
            Foo::B => {
                fooc = Foo::C;
                1
            },
            Foo::C => break,
        };
    }

    loop {
        val = match fooc {
            Foo::A => 0,
            Foo::B => 1,
            Foo::C => {
                break;
            },
        };
    }
}

macro_rules! set_or_ret {
    ($opt:expr, $a:expr) => {{
        match $opt {
            Some(val) => $a = val,
            None => return,
        }
    }};
}

fn ret_in_macro(opt: Option<u8>) {
    let opt: Option<u8> = Some(1);
    let mut a: u8 = 0;
    loop {
        set_or_ret!(opt, a);
    }

    let res: Result<bool, u8> = Ok(true);
    loop {
        match res {
            Ok(true) => set_or_ret!(opt, a),
            _ => do_something(),
        }
    }
}

fn panic_like_macros_1() {
    loop {
        do_something();
        panic!();
    }
}

fn panic_like_macros_2() {
    let mut x = 0;

    loop {
        do_something();
        if true {
            todo!();
        }
    }
    loop {
        do_something();
        x += 1;
        assert_eq!(x, 0);
    }
    loop {
        do_something();
        assert!(x % 2 == 0);
    }
    loop {
        do_something();
        match Some(1) {
            Some(n) => println!("{n}"),
            None => unreachable!("It won't happen"),
        }
    }
}

fn exit_directly(cond: bool) {
    loop {
        if cond {
            std::process::exit(0);
        }
    }
}

trait MyTrait {
    fn problematic_trait_method() {
        loop {
            //~^ ERROR: infinite loop detected
            do_something();
        }
    }
    fn could_be_problematic();
}

impl MyTrait for String {
    fn could_be_problematic() {
        loop {
            //~^ ERROR: infinite loop detected
            do_something();
        }
    }
}

fn inf_loop_in_closure() {
    let _loop_forever = || {
        loop {
            //~^ ERROR: infinite loop detected
            do_something();
        }
    };

    let _somehow_ok = || -> ! {
        loop {
            do_something();
        }
    };
}

fn inf_loop_in_res() -> Result<(), i32> {
    Ok(loop {
        do_something()
    })
}

with_span! { span
    fn no_loop() {}
}

with_span! { span
    fn with_loop() {
        loop {
            do_nothing();
        }
    }
}

fn do_nothing() {}

fn span_inside_fn() {
    with_span! { span
        loop {
            do_nothing();
        }
    }
}

fn main() {}
