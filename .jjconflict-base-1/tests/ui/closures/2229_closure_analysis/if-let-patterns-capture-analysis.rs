//@ edition:2021

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(irrefutable_let_patterns)]

enum SingleVariant {
    Pair(i32, String),
}

fn if_let_closure() {
    let variant = SingleVariant::Pair(1, "hello".into());

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        if let SingleVariant::Pair(ref n, s) = variant {
            //~^ NOTE: Capturing variant[(0, 0)] -> Immutable
            //~| NOTE: Capturing variant[(1, 0)] -> ByValue
            //~| NOTE: Min Capture variant[(0, 0)] -> Immutable
            //~| NOTE: Min Capture variant[(1, 0)] -> ByValue
            let _ = (n, s);
        }
    };

    c();
}

fn match_closure() {
    let variant = SingleVariant::Pair(1, "hello".into());

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        match variant {
            //~^ NOTE: Capturing variant[(0, 0)] -> Immutable
            //~| NOTE: Capturing variant[(1, 0)] -> ByValue
            //~| NOTE: Min Capture variant[(0, 0)] -> Immutable
            //~| NOTE: Min Capture variant[(1, 0)] -> ByValue
            SingleVariant::Pair(ref n, s) => {
                let _ = (n, s);
            }
        }
    };

    c();
}

fn main() {
    if_let_closure();
    match_closure();
}
