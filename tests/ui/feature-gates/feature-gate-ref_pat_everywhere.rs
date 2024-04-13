pub fn main() {
    if let Some(Some(&x)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
}
