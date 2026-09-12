fn main() {
    let box x = Box::new('c'); //~ ERROR `box` patterns have been removed
    let _: char = x;

    struct Packet { x: Box<i32> }

    let Packet { box x } = Packet { x: Box::new(0) }; //~ ERROR `box` patterns have been removed
    let _: i32 = x;

    let Packet { box ref mut x }; //~ ERROR `box` patterns have been removed
}
