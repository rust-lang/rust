//@ run-pass

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

// Test that default methods are cascaded correctly

// First, test only use of explicit `default` items:

trait Foo {
    fn foo(&self) -> bool;
}

// Specialization tree for Foo:
//
//        T
//       / \
//    i32   i64

impl<T> Foo for T {
    default fn foo(&self) -> bool { false }
}

impl Foo for i32 {}

impl Foo for i64 {
    fn foo(&self) -> bool { true }
}

fn test_foo() {
    assert!(!0i8.foo());
    assert!(!0i32.foo());
    assert!(0i64.foo());
}

// Next, test mixture of explicit `default` and provided methods:

trait Bar {
    fn bar(&self) -> i32 { 0 }
}

// Specialization tree for Bar.
// Uses of $ designate that method is provided
//
//           $Bar   (the trait)
//             |
//             T
//            /|\
//           / | \
//          /  |  \
//         /   |   \
//        /    |    \
//       /     |     \
//     $i32   &str  $Vec<T>
//                    /\
//                   /  \
//            Vec<i32>  $Vec<i64>

impl<T> Bar for T {
    default fn bar(&self) -> i32 { 0 }
}

impl Bar for i32 {
    fn bar(&self) -> i32 { 1 }
}
impl<'a> Bar for &'a str {}

impl<T> Bar for Vec<T> {
    default fn bar(&self) -> i32 { 2 }
}
impl Bar for Vec<i32> {}
impl Bar for Vec<i64> {
    fn bar(&self) -> i32 { 3 }
}

fn test_bar() {
    assert!(0u8.bar() == 0);
    assert!(0i32.bar() == 1);
    assert!("hello".bar() == 0);
    assert!(vec![()].bar() == 2);
    assert!(vec![0i32].bar() == 2);
    assert!(vec![0i64].bar() == 3);
}

fn main() {
    test_foo();
    test_bar();
}
