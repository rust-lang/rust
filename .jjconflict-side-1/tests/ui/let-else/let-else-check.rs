#![deny(unused_variables)]

fn main() {
    // type annotation, attributes
    #[allow(unused_variables)]
    let Some(_): Option<u32> = Some(Default::default()) else {
        let x = 1; // OK
        return;
    };

    let Some(_): Option<u32> = Some(Default::default()) else {
        let x = 1; //~ ERROR unused variable: `x`
        return;
    };

    let x = 1; //~ ERROR unused variable: `x`
}
