use std::cell::RefCell;
use std::collections::HashMap;
use std::thread_local;

#[test]
fn smoke() {
    fn square(i: i32) -> i32 {
        i * i
    }
    thread_local!(static FOO: i32 = square(3));

    FOO.with(|f| {
        assert_eq!(*f, 9);
    });
}

#[test]
fn hashmap() {
    fn map() -> RefCell<HashMap<i32, i32>> {
        let mut m = HashMap::new();
        m.insert(1, 2);
        RefCell::new(m)
    }
    thread_local!(static FOO: RefCell<HashMap<i32, i32>> = map());

    FOO.with(|map| {
        assert_eq!(map.borrow()[&1], 2);
    });
}

#[test]
fn refcell_vec() {
    thread_local!(static FOO: RefCell<Vec<u32>> = RefCell::new(vec![1, 2, 3]));

    FOO.with(|vec| {
        assert_eq!(vec.borrow().len(), 3);
        vec.borrow_mut().push(4);
        assert_eq!(vec.borrow()[3], 4);
    });
}
