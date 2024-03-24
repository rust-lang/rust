pub fn main() {
    if let Some(&x) = Some(0) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
    if let &Some(x) = &mut Some(0) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
}
