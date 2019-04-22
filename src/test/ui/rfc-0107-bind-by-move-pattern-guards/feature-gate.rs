// Check that pattern-guards with move-bound variables is only allowed
// with the appropriate set of feature gates. (Note that we require
// the code to opt into MIR-borrowck in *some* way before the feature
// will work; we use the revision system here to enumerate a number of
// ways that opt-in could occur.)

// gate-test-bind_by_move_pattern_guards

// revisions: no_gate gate_and_2015 gate_and_2018 gate_and_znll gate_and_feature_nll

// (We're already testing NLL behavior quite explicitly, no need for compare-mode=nll.)
// ignore-compare-mode-nll

#![feature(rustc_attrs)]

#![cfg_attr(gate_and_2015, feature(bind_by_move_pattern_guards))]
#![cfg_attr(gate_and_2018, feature(bind_by_move_pattern_guards))]
#![cfg_attr(gate_and_znll, feature(bind_by_move_pattern_guards))]
#![cfg_attr(gate_and_feature_nll, feature(bind_by_move_pattern_guards))]

#![cfg_attr(gate_and_feature_nll, feature(nll))]

//[gate_and_2015] edition:2015
//[gate_and_2018] edition:2018
//[gate_and_znll] compile-flags: -Z borrowck=mir

struct A { a: Box<i32> }

fn foo(n: i32) {
    let x = A { a: Box::new(n) };
    let _y = match x {

        A { a: v } if *v == 42 => v,
        //[no_gate]~^ ERROR cannot bind by-move into a pattern guard

        _ => Box::new(0)
    };
}

#[rustc_error]
fn main() {
    foo(107)
}
//[gate_and_2015]~^^^ ERROR compilation successful
//[gate_and_2018]~^^^^ ERROR compilation successful
//[gate_and_znll]~^^^^^ ERROR compilation successful
//[gate_and_feature_nll]~^^^^^^ ERROR compilation successful
