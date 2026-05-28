fn main() {
    let r: f32 = 1.0;
    match r {
        [a, b] => {
        //~^ ERROR E0529
        }
    }
}
