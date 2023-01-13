// This should not ICE
pub fn test() {
    macro_rules! foo {
        () => ()
    }
}
