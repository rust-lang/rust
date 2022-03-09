// build-pass
// compile-flags: -Zdrop-tracking
// edition:2021

enum Never {}

fn f() -> Never { todo!() }

fn main() {
    let _ = async {
        let a = String::new();
        let b = f();
        async {}.await;
        drop(a);
        drop(b);
    };
}
