//@ run-pass

fn iter_vec<T, F>(v: Vec<T>, mut f: F) where F: FnMut(&T) { for x in &v { f(x); } }

pub fn main() {
    let v = vec![1, 2, 3, 4, 5];
    let mut sum = 0;
    iter_vec(v.clone(), |i| {
        iter_vec(v.clone(), |j| {
            sum += *i * *j;
        });
    });
    println!("{}", sum);
    assert_eq!(sum, 225);
}
