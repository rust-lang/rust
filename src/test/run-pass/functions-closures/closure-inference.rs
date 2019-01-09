// run-pass


fn foo(i: isize) -> isize { i + 1 }

fn apply<A, F>(f: F, v: A) -> A where F: FnOnce(A) -> A { f(v) }

pub fn main() {
    let f = {|i| foo(i)};
    assert_eq!(apply(f, 2), 3);
}
