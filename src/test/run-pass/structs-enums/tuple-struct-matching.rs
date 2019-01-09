// run-pass
struct Foo(isize, isize);

pub fn main() {
    let x = Foo(1, 2);
    match x {
        Foo(a, b) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            println!("{} {}", a, b);
        }
    }
}
