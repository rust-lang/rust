fn f<'a: 'static>(_: &'a i32) {}

fn main() {
    let x = 0;
    f(&x); //~ERROR does not live long enough
}
