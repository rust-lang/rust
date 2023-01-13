// edition:2021

#![feature(closure_track_caller, stmt_expr_attributes)]

fn main() {
    let _ = #[track_caller] async {
        //~^ ERROR attribute should be applied to a function definition [E0739]
    };
}
