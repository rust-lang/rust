fn main() {
    let box x = Box::new('c'); //~ ERROR `box_patterns` have been removed
    let _: char = x; //~ ERROR cannot find value `x` in this scope [E0425]

    struct Packet { x: Box<i32> }

    let Packet { box x } = Packet { x: Box::new(0) }; //~ ERROR `box_patterns` have been removed
    let _: i32 = x;
}
