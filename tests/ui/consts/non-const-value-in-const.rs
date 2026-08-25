fn main() {
    let x = 5;
    const Y: i32 = x; //~ ERROR attempt to use a non-constant value in a constant [E0435]

    let x = 5;
    let _ = [0; x]; //~ ERROR attempt to use a non-constant value in a constant [E0435]
}
