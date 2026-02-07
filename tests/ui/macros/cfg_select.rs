#![crate_type = "lib"]
#![warn(unreachable_cfg_select_predicates)] // Unused warnings are disabled by default in UI tests.

fn print() {
    println!(cfg_select! {
        unix => { "unix" }
        _ => { "not unix" }
    });
}

fn print_2() {
    println!(cfg_select! {
        unix => "unix",
        _ => "not unix",
    });
}

fn arm_rhs_expr_1() -> i32 {
    cfg_select! {
        true => 1
    }
}

fn arm_rhs_expr_2() -> i32 {
    cfg_select! {
        false => 2,
        true => 1
    }
}

fn arm_rhs_expr_3() -> i32 {
    cfg_select! {
        any(true) => 1,
        any(false) => 2,
        any(true) => { 42 }
        any(true) => { 42 },
        any(false) => -1 as i32,
        any(true) => 2 + 2,
        any(false) => "",
        any(true) => if true { 42 } else { 84 }
        any(false) => if true { 42 } else { 84 },
        any(true) => return 42,
        any(false) => loop {}
        any(true) => (1, 2),
        any(false) => (1, 2,),
        any(true) => todo!(),
        any(false) => println!("hello"),
    }
}

fn expand_to_statements() -> i32 {
    cfg_select! {
        false => {
            let b = 2;
            b + 1
        }
        true => {
            let a = 1;
            a + 1
        }
    }
}

type ExpandToType = cfg_select! {
    unix => { u32 },
    _ => i32,
};

fn expand_to_pattern(x: Option<i32>) -> bool {
    match x {
        (cfg_select! {
            unix => Some(n),
            _ => None,
        }) => true,
        _ => false,
    }
}

cfg_select! {
    false => {
        fn foo() {}
    }
    _ => {
        fn bar() {}
    }
}

struct S;

impl S {
    cfg_select! {
        false => {
            fn foo() {}
        }
        _ => {
            fn bar() {}
        }
    }
}

trait T {
    cfg_select! {
        false => {
            fn a();
        }
        _ => {
            fn b();
        }
    }
}

impl T for S {
    cfg_select! {
        false => {
            fn a() {}
        },
        _ => {
            fn b() {}
        }
    }
}

extern "C" {
    cfg_select! {
        false => {
            fn puts(s: *const i8) -> i32;
        }
        _ => {
            fn printf(fmt: *const i8, ...) -> i32;
        }
    }
}

cfg_select! {
    _ => {}
    true => {}
    //~^ WARN unreachable configuration predicate
}

cfg_select! {
    true => {}
    _ => {}
    //~^ WARN unreachable configuration predicate
}

cfg_select! {
    unix => {}
    not(unix) => {},
    _ => {}
    //~^ WARN unreachable configuration predicate
}

cfg_select! {
    test => {}
    test => {}
    //~^ WARN unreachable configuration predicate
    _ => {}
    //~^ WARN unreachable configuration predicate
}

cfg_select! {
    //~^ ERROR none of the predicates in this `cfg_select` evaluated to true
    false => {}
}

cfg_select! {}
//~^ ERROR none of the predicates in this `cfg_select` evaluated to true

cfg_select! {
    => {}
    //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `=>`
}

cfg_select! {
    () => {}
    //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found expression
}

cfg_select! {
    "str" => {}
    //~^ ERROR malformed `cfg_select` macro input [E0539]
}

cfg_select! {
    a::b => {}
    //~^ ERROR malformed `cfg_select` macro input [E0539]
}

cfg_select! {
    a() => {}
    //~^ ERROR invalid predicate `a` [E0537]
}

cfg_select! {
    a + 1 => {}
    //~^ ERROR expected one of `(`, `::`, `=>`, or `=`, found `+`
    //~| WARN unexpected `cfg` condition name
}

cfg_select! {
    cfg!() => {}
    //~^ ERROR expected one of `(`, `::`, `=>`, or `=`, found `!`
    //~| WARN unexpected `cfg` condition name
}
