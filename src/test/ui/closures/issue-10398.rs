fn main() {
    let x: Box<_> = Box::new(1);
    let f = move|| {
        let _a = x;
        drop(x);
        //~^ ERROR: use of moved value: `x`
    };
    f();
}
