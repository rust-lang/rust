// run-pass

struct J { j: isize }

pub fn main() {
    let i: Box<_> = Box::new(J {
        j: 100
    });
    assert_eq!(i.j, 100);
}
