// run-pass

#![allow(non_camel_case_types)]

trait sum {
    fn sum_(self) -> isize;
}

// Note: impl on a slice
impl<'a> sum for &'a [isize] {
    fn sum_(self) -> isize {
        self.iter().fold(0, |a, &b| a + b)
    }
}

fn call_sum(x: &[isize]) -> isize { x.sum_() }

pub fn main() {
    let x = vec![1, 2, 3];
    let y = call_sum(&x);
    println!("y=={}", y);
    assert_eq!(y, 6);

    let x = vec![1, 2, 3];
    let y = x.sum_();
    println!("y=={}", y);
    assert_eq!(y, 6);

    let x = vec![1, 2, 3];
    let y = x.sum_();
    println!("y=={}", y);
    assert_eq!(y, 6);
}
