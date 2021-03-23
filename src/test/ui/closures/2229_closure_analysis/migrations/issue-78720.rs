// run-pass

#![warn(disjoint_capture_drop_reorder)]

fn main() {
    if let a = "" {
    //~^ WARNING: irrefutable `if let` pattern
        drop(|_: ()| drop(a));
    }
}
