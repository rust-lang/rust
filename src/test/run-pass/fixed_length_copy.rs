
// error on implicit copies to check fixed length vectors
// are implicitly copyable 
#[warn(err_implicit_copies)]
fn main() {
    let arr = [1,2,3]/3;
    let arr2 = arr;
    assert(arr[1] == 2);
    assert(arr2[2] == 3);
}
