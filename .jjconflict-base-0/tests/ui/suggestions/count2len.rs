fn main() {
    let slice = [1,2,3,4];
    let vec = vec![1,2,3,4];

    slice.count(); //~ERROR: E0599
    vec.count(); //~ERROR: E0599
    vec.as_slice().count(); //~ERROR: E0599
}
