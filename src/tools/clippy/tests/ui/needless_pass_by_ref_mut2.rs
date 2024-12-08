// If both `inner_async3` and `inner_async4` are present, aliases are declared after
// they're used in `inner_async4` for some reasons... This test ensures that no
// only `v` is marked as not used mutably in `inner_async4`.

#![allow(clippy::redundant_closure_call)]
#![warn(clippy::needless_pass_by_ref_mut)]

async fn inner_async3(x: &mut i32, y: &mut u32) {
    //~^ ERROR: this argument is a mutable reference, but not used mutably
    async {
        *y += 1;
    }
    .await;
}

async fn inner_async4(u: &mut i32, v: &mut u32) {
    //~^ ERROR: this argument is a mutable reference, but not used mutably
    async {
        *u += 1;
    }
    .await;
}

fn main() {}
