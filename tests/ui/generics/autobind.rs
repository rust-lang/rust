// run-pass

fn f<T>(x: Vec<T>) -> T { return x.into_iter().next().unwrap(); }

fn g<F>(act: F) -> isize where F: FnOnce(Vec<isize>) -> isize { return act(vec![1, 2, 3]); }

pub fn main() {
    assert_eq!(g(f), 1);
    let f1 = f;
    assert_eq!(f1(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
               "x".to_string());
}
