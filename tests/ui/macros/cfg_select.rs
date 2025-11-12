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
    //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `(`
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
}

cfg_select! {
    cfg!() => {}
    //~^ ERROR expected one of `(`, `::`, `=>`, or `=`, found `!`
}
