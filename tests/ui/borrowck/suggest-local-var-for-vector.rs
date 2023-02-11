fn main() {
    let mut vec = vec![0u32; 420];
    vec[vec.len() - 1] = 123; //~ ERROR cannot borrow `vec` as immutable because it is also borrowed as mutable
}
