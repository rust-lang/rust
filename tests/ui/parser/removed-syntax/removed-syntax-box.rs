//@ run-rustfix

fn main() {
    #[allow(dead_code)]
    struct T {
        a: u8,
        b: u8,
    }
    let _ = box (); //~ ERROR `box_syntax` has been removed
    let _ = box 1; //~ ERROR `box_syntax` has been removed
    let _ = box T { a: 12, b: 18 }; //~ ERROR `box_syntax` has been removed
    let _ = box [5; 30]; //~ ERROR `box_syntax` has been removed
    let _: Box<()> = box (); //~ ERROR `box_syntax` has been removed
}
