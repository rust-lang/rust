fn main() {
    // this is work
    match () {
        _ if true => || (),
        _ => (|| ()) as unsafe fn(),
        _ if true => (|| ()) as fn(),
    };

    // this is not
    match () {
        _ if true => || (), //~ ERROR cannot coerce between `fn()` and unsafe function pointers
        _ if true => (|| ()) as fn(),
        _ => (|| ()) as unsafe fn(),
    };
}
