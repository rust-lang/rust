// Check if rustc still displays the misleading hint to write `.` instead of `..`
fn main() {
    let width = 10;
    // ...
    for _ in 0..w {
        //~^ ERROR cannot find value `w`
    }
}
