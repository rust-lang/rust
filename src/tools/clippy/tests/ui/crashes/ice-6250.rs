// originally from glacier/fixed/77218.rs
// ice while adjusting...

pub struct Cache {
    data: Vec<i32>,
}

pub fn list_data(cache: &Cache, key: usize) {
    for reference in vec![1, 2, 3] {
        if
        /* let */
        Some(reference) = cache.data.get(key) {
            unimplemented!()
        }
    }
}
