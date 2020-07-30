// run-pass

struct A<T: 'static>(&'static T);
struct B<T: 'static + ?Sized> {
    x: &'static T,
}
static C: A<B<B<[u8]>>> = {
    A(&B {
        x: &B { x: b"hi" as &[u8] },
    })
};

fn main() {
    assert_eq!(b"hi", C.0.x.x);
}
