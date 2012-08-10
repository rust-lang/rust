fn main() {
    let _x = || {
        return //~ ERROR: `return` in block function
    };
}
