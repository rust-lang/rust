// Make sure we don't ICE when a foreign fn doesn't implement `Fn` due to arg mismatch.

unsafe extern "Rust" {
    pub safe fn foo();
    pub safe fn bar(x: u32);
}

fn test(_: impl Fn(i32)) {}

fn main() {
    test(foo); //~ ERROR function is expected to take 1 argument, but it takes 0 arguments
    test(bar); //~ ERROR type mismatch in function arguments
}
