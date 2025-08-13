//@ edition: 2018
// Regression test for <https://github.com/rust-lang/rust/issues/144155>.

struct NeedsDrop<'a>(&'a Vec<i32>);

async fn await_point() {}

impl Drop for NeedsDrop<'_> {
    fn drop(&mut self) {}
}

fn foo() {
    let v = vec![1, 2, 3];
    let x = NeedsDrop(&v);
    let c = async {
        std::future::ready(()).await;
        drop(x);
    };
    drop(v);
    //~^ ERROR cannot move out of `v` because it is borrowed
}

fn main() {}
