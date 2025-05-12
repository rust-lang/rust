// FIXME(tschottdorf): This should compile. See #44912.

pub fn main() {
    let x = &Some((3, 3));
    let _: &i32 = match x {
        Some((x, 3)) | &Some((ref x, 5)) => x,
        //~^ ERROR is bound inconsistently
        _ => &5i32,
    };
}
