#![feature(cfg_select)]
#![crate_type = "lib"]

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
        true => 1,
        false => 2
    }
}

fn arm_rhs_expr_3() -> i32 {
    cfg_select! {
        true => 1,
        false => 2,
        true => { 42 }
        false => -1 as i32,
        true => 2 + 2,
        false => "",
        true => if true { 42 } else { 84 }
        false => if true { 42 } else { 84 },
        true => return 42,
        false => loop {}
        true => (1, 2),
        false => (1, 2,),
        true => todo!(),
        false => println!("hello"),
    }
}

fn expand_to_statements() -> i32 {
    cfg_select! {
        true => {
            let a = 1;
            a + 1
        }
        false => {
            let b = 2;
            b + 1
        }
    }
}

type ExpandToType = cfg_select! {
    unix => u32,
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
    true => {
        fn foo() {}
    }
    _ => {
        fn bar() {}
    }
}

struct S;

impl S {
    cfg_select! {
        true => {
            fn foo() {}
        }
        _ => {
            fn bar() {}
        }
    }
}

trait T {
    cfg_select! {
        true => {
            fn a();
        }
        _ => {
            fn b();
        }
    }
}

impl T for S {
    cfg_select! {
        true => {
            fn a() {}
        }
        _ => {
            fn b() {}
        }
    }
}

extern "C" {
    cfg_select! {
        true => {
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
    //~^ WARN unreachable predicate
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
