fn main() {
    let x: &'static u32 = {
        let y = 42;
        &y //~ ERROR does not live long enough
    };
    let x: &'static u32 = &{ //~ ERROR does not live long enough
        let y = 42;
        y
    };
}
