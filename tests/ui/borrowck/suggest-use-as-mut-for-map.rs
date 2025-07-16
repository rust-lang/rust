// We don't suggest change `&` to `&mut`
// instead we suggest using .get_mut() instead of &mut, see issue #143732

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
    let mut map = std::collections::BTreeMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test"); //~ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference [E0596]

    let mut map = std::collections::HashMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test"); //~ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference [E0596]

    let mut vec = vec![String::new(), String::new()];
    let string = &vec[0];
    string.push_str("test"); //~ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference [E0596]

    // Example with our custom ReadOnlyVec type
    let read_only_vec = ReadOnlyVec::new(vec![String::new(), String::new()]);
    let string_ref = &read_only_vec[0];
    string_ref.push_str("test"); //~ ERROR cannot borrow `*string_ref` as mutable, as it is behind a `&` reference [E0596]
}
