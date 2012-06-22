// xfail-test
fn find<T>(_f: fn(@T) -> bool, _v: [@T]) {}

fn main() {
    let x = 10, arr = [];
    find({|f| f.id == x}, arr);
    arr += [{id: 20}]; // This assigns a type to arr
}
