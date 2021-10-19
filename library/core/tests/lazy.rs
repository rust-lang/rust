use core::{
    cell::Cell,
    lazy::{Lazy, OnceCell},
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

#[test]
fn once_cell() {
    let c = OnceCell::new();
    assert!(c.get().is_none());
    c.get_or_init(|| 92);
    assert_eq!(c.get(), Some(&92));

    c.get_or_init(|| panic!("Kabom!"));
    assert_eq!(c.get(), Some(&92));
}

#[test]
fn once_cell_get_mut() {
    let mut c = OnceCell::new();
    assert!(c.get_mut().is_none());
    c.set(90).unwrap();
    *c.get_mut().unwrap() += 2;
    assert_eq!(c.get_mut(), Some(&mut 92));
}

#[test]
fn once_cell_drop() {
    static DROP_CNT: AtomicUsize = AtomicUsize::new(0);
    struct Dropper;
    impl Drop for Dropper {
        fn drop(&mut self) {
            DROP_CNT.fetch_add(1, SeqCst);
        }
    }

    let x = OnceCell::new();
    x.get_or_init(|| Dropper);
    assert_eq!(DROP_CNT.load(SeqCst), 0);
    drop(x);
    assert_eq!(DROP_CNT.load(SeqCst), 1);
}

#[test]
fn unsync_once_cell_drop_empty() {
    let x = OnceCell::<&'static str>::new();
    drop(x);
}

#[test]
const fn once_cell_const() {
    let _once_cell: OnceCell<u32> = OnceCell::new();
    let _once_cell: OnceCell<u32> = OnceCell::from(32);
}

#[test]
fn clone() {
    let s = OnceCell::new();
    let c = s.clone();
    assert!(c.get().is_none());

    s.set("hello").unwrap();
    let c = s.clone();
    assert_eq!(c.get().map(|c| *c), Some("hello"));
}

#[test]
fn from_impl() {
    assert_eq!(OnceCell::from("value").get(), Some(&"value"));
    assert_ne!(OnceCell::from("foo").get(), Some(&"bar"));
}

#[test]
fn partialeq_impl() {
    assert!(OnceCell::from("value") == OnceCell::from("value"));
    assert!(OnceCell::from("foo") != OnceCell::from("bar"));

    assert!(OnceCell::<&'static str>::new() == OnceCell::new());
    assert!(OnceCell::<&'static str>::new() != OnceCell::from("value"));
}

#[test]
fn into_inner() {
    let cell: OnceCell<&'static str> = OnceCell::new();
    assert_eq!(cell.into_inner(), None);
    let cell = OnceCell::new();
    cell.set("hello").unwrap();
    assert_eq!(cell.into_inner(), Some("hello"));
}

#[test]
fn lazy_new() {
    let called = Cell::new(0);
    let x = Lazy::new(|| {
        called.set(called.get() + 1);
        92
    });

    assert_eq!(called.get(), 0);

    let y = *x - 30;
    assert_eq!(y, 62);
    assert_eq!(called.get(), 1);

    let y = *x - 30;
    assert_eq!(y, 62);
    assert_eq!(called.get(), 1);
}

#[test]
fn aliasing_in_get() {
    let x = OnceCell::new();
    x.set(42).unwrap();
    let at_x = x.get().unwrap(); // --- (shared) borrow of inner `Option<T>` --+
    let _ = x.set(27); // <-- temporary (unique) borrow of inner `Option<T>`   |
    println!("{}", at_x); // <------- up until here ---------------------------+
}

#[test]
#[should_panic(expected = "reentrant init")]
fn reentrant_init() {
    let x: OnceCell<Box<i32>> = OnceCell::new();
    let dangling_ref: Cell<Option<&i32>> = Cell::new(None);
    x.get_or_init(|| {
        let r = x.get_or_init(|| Box::new(92));
        dangling_ref.set(Some(r));
        Box::new(62)
    });
    eprintln!("use after free: {:?}", dangling_ref.get().unwrap());
}

#[test]
fn dropck() {
    let cell = OnceCell::new();
    {
        let s = String::new();
        cell.set(&s).unwrap();
    }
}
