//@ run-pass
//@ needs-unwind

#![allow(dead_code, unreachable_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::panic::{self, AssertUnwindSafe, UnwindSafe};

// This struct is used to record the order in which elements are dropped
struct PushOnDrop {
    vec: Rc<RefCell<Vec<u32>>>,
    val: u32
}

impl PushOnDrop {
    fn new(val: u32, vec: Rc<RefCell<Vec<u32>>>) -> PushOnDrop {
        PushOnDrop { vec, val }
    }
}

impl Drop for PushOnDrop {
    fn drop(&mut self) {
        self.vec.borrow_mut().push(self.val)
    }
}

impl UnwindSafe for PushOnDrop { }

// Structs
struct TestStruct {
    x: PushOnDrop,
    y: PushOnDrop,
    z: PushOnDrop
}

// Tuple structs
struct TestTupleStruct(PushOnDrop, PushOnDrop, PushOnDrop);

// Enum variants
enum TestEnum {
    Tuple(PushOnDrop, PushOnDrop, PushOnDrop),
    Struct { x: PushOnDrop, y: PushOnDrop, z: PushOnDrop }
}

fn test_drop_tuple() {
    // Tuple fields are dropped in the same order they are declared
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let test_tuple = (PushOnDrop::new(1, dropped_fields.clone()),
                      PushOnDrop::new(2, dropped_fields.clone()));
    drop(test_tuple);
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // Panic during construction means that fields are treated as local variables
    // Therefore they are dropped in reverse order of initialization
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        (PushOnDrop::new(2, cloned.clone()),
         PushOnDrop::new(1, cloned.clone()),
         panic!("this panic is caught :D"));
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);
}

fn test_drop_struct() {
    // Struct fields are dropped in the same order they are declared
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let test_struct = TestStruct {
        x: PushOnDrop::new(1, dropped_fields.clone()),
        y: PushOnDrop::new(2, dropped_fields.clone()),
        z: PushOnDrop::new(3, dropped_fields.clone()),
    };
    drop(test_struct);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // The same holds for tuple structs
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let test_tuple_struct = TestTupleStruct(PushOnDrop::new(1, dropped_fields.clone()),
                                            PushOnDrop::new(2, dropped_fields.clone()),
                                            PushOnDrop::new(3, dropped_fields.clone()));
    drop(test_tuple_struct);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // Panic during struct construction means that fields are treated as local variables
    // Therefore they are dropped in reverse order of initialization
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestStruct {
            x: PushOnDrop::new(2, cloned.clone()),
            y: PushOnDrop::new(1, cloned.clone()),
            z: panic!("this panic is caught :D")
        };
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // Test with different initialization order
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestStruct {
            y: PushOnDrop::new(2, cloned.clone()),
            x: PushOnDrop::new(1, cloned.clone()),
            z: panic!("this panic is caught :D")
        };
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // The same holds for tuple structs
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestTupleStruct(PushOnDrop::new(2, cloned.clone()),
                        PushOnDrop::new(1, cloned.clone()),
                        panic!("this panic is caught :D"));
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);
}

fn test_drop_enum() {
    // Enum variants are dropped in the same order they are declared
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let test_struct_enum = TestEnum::Struct {
        x: PushOnDrop::new(1, dropped_fields.clone()),
        y: PushOnDrop::new(2, dropped_fields.clone()),
        z: PushOnDrop::new(3, dropped_fields.clone())
    };
    drop(test_struct_enum);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // The same holds for tuple enum variants
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let test_tuple_enum = TestEnum::Tuple(PushOnDrop::new(1, dropped_fields.clone()),
                                          PushOnDrop::new(2, dropped_fields.clone()),
                                          PushOnDrop::new(3, dropped_fields.clone()));
    drop(test_tuple_enum);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // Panic during enum construction means that fields are treated as local variables
    // Therefore they are dropped in reverse order of initialization
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestEnum::Struct {
            x: PushOnDrop::new(2, cloned.clone()),
            y: PushOnDrop::new(1, cloned.clone()),
            z: panic!("this panic is caught :D")
        };
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // Test with different initialization order
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestEnum::Struct {
            y: PushOnDrop::new(2, cloned.clone()),
            x: PushOnDrop::new(1, cloned.clone()),
            z: panic!("this panic is caught :D")
        };
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // The same holds for tuple enum variants
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        TestEnum::Tuple(PushOnDrop::new(2, cloned.clone()),
                        PushOnDrop::new(1, cloned.clone()),
                        panic!("this panic is caught :D"));
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);
}

fn test_drop_list() {
    // Elements in a Vec are dropped in the same order they are pushed
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let xs = vec![PushOnDrop::new(1, dropped_fields.clone()),
                  PushOnDrop::new(2, dropped_fields.clone()),
                  PushOnDrop::new(3, dropped_fields.clone())];
    drop(xs);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // The same holds for arrays
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let xs = [PushOnDrop::new(1, dropped_fields.clone()),
              PushOnDrop::new(2, dropped_fields.clone()),
              PushOnDrop::new(3, dropped_fields.clone())];
    drop(xs);
    assert_eq!(*dropped_fields.borrow(), &[1, 2, 3]);

    // Panic during vec construction means that fields are treated as local variables
    // Therefore they are dropped in reverse order of initialization
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        vec![
            PushOnDrop::new(2, cloned.clone()),
            PushOnDrop::new(1, cloned.clone()),
            panic!("this panic is caught :D")
        ];
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);

    // The same holds for arrays
    let dropped_fields = Rc::new(RefCell::new(Vec::new()));
    let cloned = AssertUnwindSafe(dropped_fields.clone());
    panic::catch_unwind(|| {
        [
            PushOnDrop::new(2, cloned.clone()),
            PushOnDrop::new(1, cloned.clone()),
            panic!("this panic is caught :D")
        ];
    }).err().unwrap();
    assert_eq!(*dropped_fields.borrow(), &[1, 2]);
}

fn main() {
    test_drop_tuple();
    test_drop_struct();
    test_drop_enum();
    test_drop_list();
}
