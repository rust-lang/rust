fn main() {
    let x = 1;
    let y = &mut x; //~ ERROR [E0596]
}
