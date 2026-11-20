//@ compile-flags: -Zmir-opt-level=0
//@ edition: 2024
//@ skip-filecheck
//@ needs-unwind

#![feature(explicit_tail_calls)]
#![feature(coroutine_trait)]
#![feature(coroutines)]
#![feature(stmt_expr_attributes)]

fn take_string(_s: String) {}

// EMIT_MIR moved_drops.repeat_move.built.after.mir
fn repeat_move() {
    let _arr: [String; 1] = [String::new(); 1];
}

// EMIT_MIR moved_drops.zero_repeat_move.built.after.mir
fn zero_repeat_move() {
    let _arr: [String; 0] = [String::new(); 0];
}

// EMIT_MIR moved_drops.array_aggregate.built.after.mir
fn array_aggregate() {
    let _arr = [String::new(), String::new()];
}

// EMIT_MIR moved_drops.tuple_aggregate.built.after.mir
fn tuple_aggregate() {
    let _tuple = (String::new(), String::new());
}

// EMIT_MIR moved_drops.adt_aggregate.built.after.mir
fn adt_aggregate() {
    struct S(String);
    let _s = S(String::new());
}

// EMIT_MIR moved_drops.closure_upvar.built.after.mir
fn closure_upvar() {
    let x = String::new();
    let _c = move || x;
}

// EMIT_MIR moved_drops.var_ref_move.built.after.mir
fn var_ref_move() {
    let x = String::new();
    let _y = x;
}

// EMIT_MIR moved_drops.call_args.built.after.mir
fn call_args() {
    take_string(String::new());
}

// EMIT_MIR moved_drops.tail_call_become.built.after.mir
fn tail_call_become(_: String) {
    become tail_call_become(String::new());
}

// EMIT_MIR moved_drops.yield_value-{closure#0}.built.after.mir
fn yield_value() {
    let _ = #[coroutine]
    || {
        yield String::new();
    };
}

fn main() {}
