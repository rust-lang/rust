// run-pass
#![feature(fn_traits, test)]

extern crate test;

fn test1(a: isize, b: (i32, i32), c: &[i32]) -> (isize, (i32, i32), &[i32]) {
    // Test passing a number of arguments including a fat pointer.
    // Also returning via an out pointer
    fn callee(a: isize, b: (i32, i32), c: &[i32]) -> (isize, (i32, i32), &[i32]) {
        (a, b, c)
    }
    callee(a, b, c)
}

fn test2(a: isize) -> isize {
    // Test passing a single argument.
    // Not using out pointer.
    fn callee(a: isize) -> isize {
        a
    }
    callee(a)
}

#[derive(PartialEq, Eq, Debug)]
struct Foo;
impl Foo {
    fn inherent_method(&self, a: isize) -> isize { a }
}

fn test3(x: &Foo, a: isize) -> isize {
    // Test calling inherent method
    x.inherent_method(a)
}

trait Bar {
    fn extension_method(&self, a: isize) -> isize { a }
}
impl Bar for Foo {}

fn test4(x: &Foo, a: isize) -> isize {
    // Test calling extension method
    x.extension_method(a)
}

fn test5(x: &dyn Bar, a: isize) -> isize {
    // Test calling method on trait object
    x.extension_method(a)
}

fn test6<T: Bar>(x: &T, a: isize) -> isize {
    // Test calling extension method on generic callee
    x.extension_method(a)
}

trait One<T = Self> {
    fn one() -> T;
}
impl One for isize {
    fn one() -> isize { 1 }
}

fn test7() -> isize {
    // Test calling trait static method
    <isize as One>::one()
}

struct Two;
impl Two {
    fn two() -> isize { 2 }
}

fn test8() -> isize {
    // Test calling impl static method
    Two::two()
}

#[allow(improper_ctypes_definitions)]
extern "C" fn simple_extern(x: u32, y: (u32, u32)) -> u32 {
    x + y.0 * y.1
}

fn test9() -> u32 {
    simple_extern(41, (42, 43))
}

fn test_closure<F>(f: &F, x: i32, y: i32) -> i32
    where F: Fn(i32, i32) -> i32
{
    f(x, y)
}

fn test_fn_object(f: &dyn Fn(i32, i32) -> i32, x: i32, y: i32) -> i32 {
    f(x, y)
}

fn test_fn_impl(f: &&dyn Fn(i32, i32) -> i32, x: i32, y: i32) -> i32 {
    // This call goes through the Fn implementation for &Fn provided in
    // core::ops::impls. It expands to a static Fn::call() that calls the
    // Fn::call() implementation of the object shim underneath.
    f(x, y)
}

fn test_fn_direct_call<F>(f: &F, x: i32, y: i32) -> i32
    where F: Fn(i32, i32) -> i32
{
    f.call((x, y))
}

fn test_fn_const_call<F>(f: &F) -> i32
    where F: Fn(i32, i32) -> i32
{
    f.call((100, -1))
}

fn test_fn_nil_call<F>(f: &F) -> i32
    where F: Fn() -> i32
{
    f()
}

fn test_fn_transmute_zst(x: ()) -> [(); 1] {
    fn id<T>(x: T) -> T {x}

    id(unsafe {
        std::mem::transmute(x)
    })
}

fn test_fn_ignored_pair() -> ((), ()) {
    ((), ())
}

fn test_fn_ignored_pair_0() {
    test_fn_ignored_pair().0
}

fn id<T>(x: T) -> T { x }

fn ignored_pair_named() -> (Foo, Foo) {
    (Foo, Foo)
}

fn test_fn_ignored_pair_named() -> (Foo, Foo) {
    id(ignored_pair_named())
}

fn test_fn_nested_pair(x: &((f32, f32), u32)) -> (f32, f32) {
    let y = *x;
    let z = y.0;
    (z.0, z.1)
}

fn test_fn_const_arg_by_ref(mut a: [u64; 4]) -> u64 {
    // Mutate the by-reference argument, which won't work with
    // a non-immediate constant unless it's copied to the stack.
    let a = test::black_box(&mut a);
    a[0] += a[1];
    a[0] += a[2];
    a[0] += a[3];
    a[0]
}

fn main() {
    assert_eq!(test1(1, (2, 3), &[4, 5, 6]), (1, (2, 3), &[4, 5, 6][..]));
    assert_eq!(test2(98), 98);
    assert_eq!(test3(&Foo, 42), 42);
    assert_eq!(test4(&Foo, 970), 970);
    assert_eq!(test5(&Foo, 8576), 8576);
    assert_eq!(test6(&Foo, 12367), 12367);
    assert_eq!(test7(), 1);
    assert_eq!(test8(), 2);
    assert_eq!(test9(), 41 + 42 * 43);

    let r = 3;
    let closure = |x: i32, y: i32| { r*(x + (y*2)) };
    assert_eq!(test_fn_const_call(&closure), 294);
    assert_eq!(test_closure(&closure, 100, 1), 306);
    let function_object = &closure as &dyn Fn(i32, i32) -> i32;
    assert_eq!(test_fn_object(function_object, 100, 2), 312);
    assert_eq!(test_fn_impl(&function_object, 100, 3), 318);
    assert_eq!(test_fn_direct_call(&closure, 100, 4), 324);

    assert_eq!(test_fn_nil_call(&(|| 42)), 42);
    assert_eq!(test_fn_transmute_zst(()), [()]);

    assert_eq!(test_fn_ignored_pair_0(), ());
    assert_eq!(test_fn_ignored_pair_named(), (Foo, Foo));
    assert_eq!(test_fn_nested_pair(&((1.0, 2.0), 0)), (1.0, 2.0));

    const ARRAY: [u64; 4] = [1, 2, 3, 4];
    assert_eq!(test_fn_const_arg_by_ref(ARRAY), 1 + 2 + 3 + 4);
}
