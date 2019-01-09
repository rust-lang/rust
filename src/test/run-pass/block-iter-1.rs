fn iter_vec<T, F>(v: Vec<T> , mut f: F) where F: FnMut(&T) { for x in &v { f(x); } }

pub fn main() {
    let v = vec![1, 2, 3, 4, 5, 6, 7];
    let mut odds = 0;
    iter_vec(v, |i| {
        if *i % 2 == 1 {
            odds += 1;
        }
    });
    println!("{}", odds);
    assert_eq!(odds, 4);
}
