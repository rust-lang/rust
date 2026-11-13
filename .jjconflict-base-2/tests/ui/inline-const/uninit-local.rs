fn main() {
    let _my_usize = const {
        let x: bool;
        while x {} //~ ERROR: `x` isn't initialized
    };
}
