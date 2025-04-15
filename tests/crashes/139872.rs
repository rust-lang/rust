//@ known-bug: #139872
//@ only-x86_64
//@ ignore-x86_64-apple-darwin
pub fn main() {
    enum A {
        B(u32),
    }
    static C: (A, u16, str);
    fn d() {
        let (_, e, _) = C;
    }
}
