use std::borrow::Cow;

// check that Cow<'a, str> implements addition
#[test]
fn check_cow_add_cow() {
    let borrowed1 = Cow::Borrowed("Hello, ");
    let borrowed2 = Cow::Borrowed("World!");
    let borrow_empty = Cow::Borrowed("");

    let owned1: Cow<'_, str> = Cow::Owned(String::from("Hi, "));
    let owned2: Cow<'_, str> = Cow::Owned(String::from("Rustaceans!"));
    let owned_empty: Cow<'_, str> = Cow::Owned(String::new());

    assert_eq!("Hello, World!", borrowed1.clone() + borrowed2.clone());
    assert_eq!("Hello, Rustaceans!", borrowed1.clone() + owned2.clone());

    assert_eq!("Hi, World!", owned1.clone() + borrowed2.clone());
    assert_eq!("Hi, Rustaceans!", owned1.clone() + owned2.clone());

    if let Cow::Owned(_) = borrowed1.clone() + borrow_empty.clone() {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    if let Cow::Owned(_) = borrow_empty.clone() + borrowed1.clone() {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    if let Cow::Owned(_) = borrowed1.clone() + owned_empty.clone() {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    if let Cow::Owned(_) = owned_empty.clone() + borrowed1.clone() {
        panic!("Adding empty strings to a borrow should note allocate");
    }
}

#[test]
fn check_cow_add_str() {
    let borrowed = Cow::Borrowed("Hello, ");
    let borrow_empty = Cow::Borrowed("");

    let owned: Cow<'_, str> = Cow::Owned(String::from("Hi, "));
    let owned_empty: Cow<'_, str> = Cow::Owned(String::new());

    assert_eq!("Hello, World!", borrowed.clone() + "World!");

    assert_eq!("Hi, World!", owned.clone() + "World!");

    if let Cow::Owned(_) = borrowed.clone() + "" {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    if let Cow::Owned(_) = borrow_empty.clone() + "Hello, " {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    if let Cow::Owned(_) = owned_empty.clone() + "Hello, " {
        panic!("Adding empty strings to a borrow should note allocate");
    }
}

#[test]
fn check_cow_add_assign_cow() {
    let mut borrowed1 = Cow::Borrowed("Hello, ");
    let borrowed2 = Cow::Borrowed("World!");
    let borrow_empty = Cow::Borrowed("");

    let mut owned1: Cow<'_, str> = Cow::Owned(String::from("Hi, "));
    let owned2: Cow<'_, str> = Cow::Owned(String::from("Rustaceans!"));
    let owned_empty: Cow<'_, str> = Cow::Owned(String::new());

    let mut s = borrowed1.clone();
    s += borrow_empty.clone();
    assert_eq!("Hello, ", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    let mut s = borrow_empty.clone();
    s += borrowed1.clone();
    assert_eq!("Hello, ", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    let mut s = borrowed1.clone();
    s += owned_empty.clone();
    assert_eq!("Hello, ", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    let mut s = owned_empty.clone();
    s += borrowed1.clone();
    assert_eq!("Hello, ", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }

    owned1 += borrowed2;
    borrowed1 += owned2;

    assert_eq!("Hi, World!", owned1);
    assert_eq!("Hello, Rustaceans!", borrowed1);
}

#[test]
fn check_cow_add_assign_str() {
    let mut borrowed = Cow::Borrowed("Hello, ");
    let borrow_empty = Cow::Borrowed("");

    let mut owned: Cow<'_, str> = Cow::Owned(String::from("Hi, "));
    let owned_empty: Cow<'_, str> = Cow::Owned(String::new());

    let mut s = borrowed.clone();
    s += "";
    assert_eq!("Hello, ", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    let mut s = borrow_empty.clone();
    s += "World!";
    assert_eq!("World!", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }
    let mut s = owned_empty.clone();
    s += "World!";
    assert_eq!("World!", s);
    if let Cow::Owned(_) = s {
        panic!("Adding empty strings to a borrow should note allocate");
    }

    owned += "World!";
    borrowed += "World!";

    assert_eq!("Hi, World!", owned);
    assert_eq!("Hello, World!", borrowed);
}

#[test]
fn check_cow_clone_from() {
    let mut c1: Cow<'_, str> = Cow::Owned(String::with_capacity(25));
    let s: String = "hi".to_string();
    assert!(s.capacity() < 25);
    let c2: Cow<'_, str> = Cow::Owned(s);
    c1.clone_from(&c2);
    assert!(c1.into_owned().capacity() >= 25);
    let mut c3: Cow<'_, str> = Cow::Borrowed("bye");
    c3.clone_from(&c2);
    assert_eq!(c2, c3);
}
