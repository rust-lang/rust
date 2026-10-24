//@ run-pass

fn grow(v: &mut Vec<isize> ) {
    v.push(1);
}

pub fn main() {
    let mut v: Vec<isize> = Vec::new();
    grow(&mut v);
    grow(&mut v);
    grow(&mut v);
    let len = v.len();
    println!("{}", len);
    assert_eq!(len, 3 as usize);
}
