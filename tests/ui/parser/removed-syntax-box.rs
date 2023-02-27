fn main() {
    struct T {
        a: u8,
        b: u8,
    }
    let _ = box () //~ ERROR expected expression, found reserved keyword `box`
    let _ = box 1;
    let _ = box T { a: 12, b: 18 };
    let _ = box [5; 30];
}
