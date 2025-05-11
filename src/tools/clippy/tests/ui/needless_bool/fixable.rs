#![warn(clippy::needless_bool)]
#![allow(
    unused,
    dead_code,
    clippy::no_effect,
    clippy::if_same_then_else,
    clippy::equatable_if_let,
    clippy::needless_if,
    clippy::needless_return,
    clippy::self_named_constructors,
    clippy::struct_field_names
)]

use std::cell::Cell;

macro_rules! bool_comparison_trigger {
    ($($i:ident: $def:expr, $stb:expr );+  $(;)*) => (

        #[derive(Clone)]
        pub struct Trigger {
            $($i: (Cell<bool>, bool, bool)),+
        }

        #[allow(dead_code)]
        impl Trigger {
            pub fn trigger(&self, key: &str) -> bool {
                $(
                    if let stringify!($i) = key {
                        return self.$i.1 && self.$i.2 == $def;
                    }
                 )+
                false
            }
        }
    )
}

fn main() {
    let x = true;
    let y = false;
    if x {
        true
    } else {
        false
    };
    //~^^^^^ needless_bool
    if x {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if x && y {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    let a = 0;
    let b = 1;

    if a == b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if a != b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if a < b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if a <= b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if a > b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if a >= b {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
    if x {
        x
    } else {
        false
    }; // would also be questionable, but we don't catch this yet
    bool_ret3(x);
    bool_ret4(x);
    bool_ret5(x, x);
    bool_ret6(x, x);
    needless_bool(x);
    needless_bool2(x);
    needless_bool3(x);
    needless_bool_condition();

    if a == b {
        true
    } else {
        // Do not lint as this comment might be important
        false
    };
}

fn bool_ret3(x: bool) -> bool {
    if x {
        return true;
    } else {
        return false;
    };
    //~^^^^^ needless_bool
}

fn bool_ret4(x: bool) -> bool {
    if x {
        return false;
    } else {
        return true;
    };
    //~^^^^^ needless_bool
}

fn bool_ret5(x: bool, y: bool) -> bool {
    if x && y {
        return true;
    } else {
        return false;
    };
    //~^^^^^ needless_bool
}

fn bool_ret6(x: bool, y: bool) -> bool {
    if x && y {
        return false;
    } else {
        return true;
    };
    //~^^^^^ needless_bool
}

fn needless_bool(x: bool) {
    if x == true {};
    //~^ bool_comparison
}

fn needless_bool2(x: bool) {
    if x == false {};
    //~^ bool_comparison
}

fn needless_bool3(x: bool) {
    bool_comparison_trigger! {
        test_one:   false, false;
        test_three: false, false;
        test_two:   true, true;
    }

    if x == true {};
    //~^ bool_comparison
    if x == false {};
    //~^ bool_comparison
}

fn needless_bool_in_the_suggestion_wraps_the_predicate_of_if_else_statement_in_brackets() {
    let b = false;
    let returns_bool = || false;

    let x = if b {
        true
    } else if returns_bool() {
        false
    } else {
        true
    };
    //~^^^^^ needless_bool
}

unsafe fn no(v: u8) -> u8 {
    v
}

#[allow(clippy::unnecessary_operation)]
fn needless_bool_condition() -> bool {
    if unsafe { no(4) } & 1 != 0 {
        true
    } else {
        false
    };
    //~^^^^^ needless_bool
    let _brackets_unneeded = if unsafe { no(4) } & 1 != 0 { true } else { false };
    //~^ needless_bool
    fn foo() -> bool {
        // parentheses are needed here
        if unsafe { no(4) } & 1 != 0 { true } else { false }
        //~^ needless_bool
    }

    foo()
}

fn issue12846() {
    let a = true;
    let b = false;

    // parentheses are needed here
    let _x = if a && b { true } else { false }.then(|| todo!());
    //~^ needless_bool
    let _x = if a && b { true } else { false } as u8;
    //~^ needless_bool

    // parentheses are not needed here
    let _x = if a { true } else { false }.then(|| todo!());
    //~^ needless_bool
}
