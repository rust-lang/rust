fn main() {
    let box x = Box::new('c'); //~ ERROR box pattern syntax is experimental
    let _: char = x;

    struct Packet { x: Box<i32> }

    let Packet { box x } = Packet { x: Box::new(0) }; //~ ERROR box pattern syntax is experimental
    let _: i32 = x;
}
