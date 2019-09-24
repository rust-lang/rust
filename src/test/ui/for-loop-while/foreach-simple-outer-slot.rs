// run-pass



pub fn main() {
    let mut sum: isize = 0;
    first_ten(|i| { println!("main"); println!("{}", i); sum = sum + i; });
    println!("sum");
    println!("{}", sum);
    assert_eq!(sum, 45);
}

fn first_ten<F>(mut it: F) where F: FnMut(isize) {
    let mut i: isize = 0;
    while i < 10 { println!("first_ten"); it(i); i = i + 1; }
}
