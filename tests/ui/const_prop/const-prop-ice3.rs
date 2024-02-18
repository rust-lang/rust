//@ run-pass (ensure that const-prop is run)

struct A<T: ?Sized>(T);

fn main() {
    let _x = &(&A([2, 3]) as &A<[i32]>).0 as *const [i32] as *const i32;
}
