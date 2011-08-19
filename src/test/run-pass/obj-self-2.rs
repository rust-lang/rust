

fn main() {
    obj foo() {
        fn m1(i: int) { i += 1; log "hi!"; }
        fn m2(i: int) { i += 1; self.m1(i); }
    }
    let a = foo();
    let i: int = 0;
    a.m1(i);
    a.m2(i);
}
