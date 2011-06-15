

fn main() {
    obj foo() {
        fn m1(int i) { i += 1; log "hi!"; }
        fn m2(int i) { i += 1; self.m1(i); }
    }
    auto a = foo();
    let int i = 0;
    a.m1(i);
    a.m2(i);
}