#![allow(clippy::redundant_clone)]
#![feature(custom_inner_attributes)]

fn main() {}

#[clippy::msrv = "1.42.0"]
fn just_under_msrv() {
    let log2_10 = 3.321928094887362;
}

#[clippy::msrv = "1.43.0"]
fn meets_msrv() {
    let log2_10 = 3.321928094887362;
    //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found
}

#[clippy::msrv = "1.44.0"]
fn just_above_msrv() {
    let log2_10 = 3.321928094887362;
    //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found
}

#[clippy::msrv = "1.42"]
fn no_patch_under() {
    let log2_10 = 3.321928094887362;
}

#[clippy::msrv = "1.43"]
fn no_patch_meets() {
    let log2_10 = 3.321928094887362;
    //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found
}

fn inner_attr_under() {
    #![clippy::msrv = "1.42"]
    let log2_10 = 3.321928094887362;
}

fn inner_attr_meets() {
    #![clippy::msrv = "1.43"]
    let log2_10 = 3.321928094887362;
    //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found
}

// https://github.com/rust-lang/rust-clippy/issues/6920
fn scoping() {
    mod m {
        #![clippy::msrv = "1.42.0"]
    }

    // Should warn
    let log2_10 = 3.321928094887362;
    //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found

    mod a {
        #![clippy::msrv = "1.42.0"]

        fn should_warn() {
            #![clippy::msrv = "1.43.0"]
            let log2_10 = 3.321928094887362;
            //~^ ERROR: approximate value of `f{32, 64}::consts::LOG2_10` found
        }

        fn should_not_warn() {
            let log2_10 = 3.321928094887362;
        }
    }
}
