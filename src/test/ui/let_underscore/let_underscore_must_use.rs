// run-pass

#[must_use]
struct MustUseType;

#[must_use]
fn must_use_function() -> () {}

fn main() {
    let _ = MustUseType; //~WARNING non-binding let on a expression marked `must_use`
    let _ = must_use_function(); //~WARNING non-binding let on a expression marked `must_use`
}
