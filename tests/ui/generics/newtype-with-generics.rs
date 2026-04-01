//! Test newtype pattern with generic parameters.

//@ run-pass

#[derive(Clone)]
struct MyVec<T>(Vec<T>);

fn extract_inner_vec<T: Clone>(wrapper: MyVec<T>) -> Vec<T> {
    let MyVec(inner_vec) = wrapper;
    inner_vec.clone()
}

fn get_first_element<T>(wrapper: MyVec<T>) -> T {
    let MyVec(inner_vec) = wrapper;
    inner_vec.into_iter().next().unwrap()
}

pub fn main() {
    let my_vec = MyVec(vec![1, 2, 3]);
    let cloned_vec = my_vec.clone();

    // Test extracting inner vector
    let extracted = extract_inner_vec(cloned_vec);
    assert_eq!(extracted[1], 2);

    // Test getting first element
    assert_eq!(get_first_element(my_vec.clone()), 1);

    // Test direct destructuring
    let MyVec(inner) = my_vec;
    assert_eq!(inner[2], 3);
}
