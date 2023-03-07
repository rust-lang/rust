// pp-exact:issue-31073.pp

fn main() {
    fn f1(x: i32, y: i32) -> i32 { y }
    let f: fn(_, i32) -> i32 = f1;
    f(1, 2);
}
