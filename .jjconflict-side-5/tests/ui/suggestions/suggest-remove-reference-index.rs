//@ run-rustfix

fn main() {
    let arr = [false];
    let i = 0usize;

    println!("{}", arr[&i]); //~ ERROR the type `[bool]` cannot be indexed by `&usize`
    println!("{}", arr[&(i + 0)]); //~ ERROR the type `[bool]` cannot be indexed by `&usize`
    println!("{}", arr[& i]); //~ ERROR the type `[bool]` cannot be indexed by `&usize`
}
