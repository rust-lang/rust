use std::collections::HashMap;





fn main() {
    let tmp: Box<_>;
    let mut buggy_map: HashMap<usize, &usize> = HashMap::new();
    buggy_map.insert(42, &*Box::new(1)); //~ ERROR temporary value dropped while borrowed

    // but it is ok if we use a temporary
    tmp = Box::new(2);
    buggy_map.insert(43, &*tmp);
}
