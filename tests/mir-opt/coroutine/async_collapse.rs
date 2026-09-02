// This test makes sure that the collapse_yields MIR pass eliminates
// identical yields so the statemachine will be smaller

//@ edition:2018

#![crate_type = "lib"]

async fn a(arg: i32) -> i32 {
    arg
}

pub async fn b(val: bool) {
    // Check there's only one suspension point, not two

    // CHECK-LABEL: fn b::{closure#0}(
    // CHECK-COUNT-1: Suspend
    // CHECK-NOT: Suspend

    if val {
        a(1).await;
    } else {
        a(-1).await;
    }
}

pub async fn c(val: bool) {
    // Check there's only three suspension points, not four

    // CHECK-LABEL: fn c::{closure#0}(
    // CHECK-COUNT-3: Suspend
    // CHECK-NOT: Suspend

    b(val).await;

    if val {
        a(1).await;
    } else {
        a(-1).await;
    }

    b(val).await;
}

enum ManyOptions {
    A,
    B,
    C,
    D,
}

pub async fn d(val: ManyOptions) {
    // Check there's only one suspension points, not four

    // CHECK-LABEL: fn d::{closure#0}(
    // CHECK-COUNT-1: Suspend
    // CHECK-NOT: Suspend

    match val {
        ManyOptions::A => {
            a(0).await;
        }
        ManyOptions::B => {
            println!("It's B!"); // Doing diverging work before the await shouldn't matter
            a(1).await;
        }
        ManyOptions::C => {
            a(2).await;
        }
        ManyOptions::D => {
            a(3).await;
        }
    }
}
pub async fn e(val: ManyOptions) {
    // Check there's only two suspension points, not four

    // CHECK-LABEL: fn e::{closure#0}(
    // CHECK-COUNT-2: Suspend
    // CHECK-NOT: Suspend

    match val {
        ManyOptions::A => {
            a(0).await;
        }
        ManyOptions::B => {
            a(1).await;
            println!("It's B!");
            // Doing diverging work after the await *does* matter.
            // Nonetheless, the others should still be optimized.
        }
        ManyOptions::C => {
            a(2).await;
        }
        ManyOptions::D => {
            a(3).await;
        }
    }
}
