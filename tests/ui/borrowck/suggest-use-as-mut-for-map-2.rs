// We don't suggest anything, see issue #143732

// Custom type that implements Index<usize> but not IndexMut
struct ReadOnlyVec<T> {
    data: Vec<T>,
}

impl<T> ReadOnlyVec<T> {
    fn new(data: Vec<T>) -> Self {
        ReadOnlyVec { data }
    }
}

impl<T> std::ops::Index<usize> for ReadOnlyVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

fn main() {
    // Example with our custom ReadOnlyVec type
    let read_only_vec = ReadOnlyVec::new(vec![String::new(), String::new()]);
    let string_ref = &read_only_vec[0];
    string_ref.push_str("test"); //~ ERROR cannot borrow `*string_ref` as mutable, as it is behind a `&` reference [E0596]
}
