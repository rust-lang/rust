// run-pass

fn takes_two(x: &isize, y: &isize) -> isize { *x + *y }

fn has_two<'a,'b>(x: &'a isize, y: &'b isize) -> isize {
    takes_two(x, y)
}

pub fn main() {
    assert_eq!(has_two(&20, &2), 22);
}
