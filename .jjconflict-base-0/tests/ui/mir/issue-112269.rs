pub fn main() {
    const x: i32 = 4;
    let x: i32 = 3;
    //~^ ERROR refutable pattern in local binding

    const y: i32 = 3;
    let y = 4;
    //~^ ERROR refutable pattern in local binding
}
