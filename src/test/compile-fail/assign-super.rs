fn main() {
    let mut x: [mut int] = [mut 3];
    let y: [int] = [3];
    x = y; //! ERROR values differ in mutability
}