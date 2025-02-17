//@ edition: 2024
// gate-test-ref_pat_eat_one_layer_2024_structural

pub fn main() {
    if let Some(Some(&x)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(Some(&0)) {
        let _: &u32 = x;
        //~^ ERROR: mismatched types
    }
    if let Some(Some(&&x)) = &Some(Some(&0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(&mut Some(&x)) = &Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
    if let Some(&Some(&mut x)) = &mut Some(&Some(0)) {
        //~^ ERROR: mismatched types
        let _: u32 = x;
    }
}
