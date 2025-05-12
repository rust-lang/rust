struct What<W = usize, X = Vec<W>>(W, X);

fn main() {
    let mut b: What<usize> = What(5, vec![1, 2, 3]);
    let c: What<usize, String> = What(1, String::from("meow"));
    b = c; //~ ERROR mismatched types

    let mut f: What<usize, Vec<String>> = What(1, vec![String::from("meow")]);
    let e: What<usize> = What(5, vec![1, 2, 3]);
    f = e; //~ ERROR mismatched types
}
