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
