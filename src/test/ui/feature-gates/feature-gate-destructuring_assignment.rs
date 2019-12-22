fn main() {
    let (a, b) = (0, 1);
    (a, b) = (2, 3); //~ ERROR destructuring assignments are unstable
}
