// rust-lang/rust#45696: This test is checking that we can return
// mutable borrows owned by boxes even when the boxes are dropped.

// run-pass

// This function shows quite directly what is going on: We have a
// reborrow of contents within the box.
fn return_borrow_from_dropped_box_1(x: Box<&mut u32>) -> &mut u32 { &mut **x }

// This function is the way you'll probably see this in practice (the
// reborrow is now implicit).
fn return_borrow_from_dropped_box_2(x: Box<&mut u32>) -> &mut u32 { *x }

// For the remaining tests we just add some fields or other
// indirection to ensure that the compiler isn't just special-casing
// the above `Box<&mut T>` as the only type that would work.

// Here we add a tuple of indirection between the box and the
// reference.
type BoxedTup<'a, 'b> = Box<(&'a mut u32, &'b mut u32)>;

fn return_borrow_of_field_from_dropped_box_1<'a>(x: BoxedTup<'a, '_>) -> &'a mut u32 {
    &mut *x.0
}

fn return_borrow_of_field_from_dropped_box_2<'a>(x: BoxedTup<'a, '_>) -> &'a mut u32 {
    x.0
}

fn return_borrow_from_dropped_tupled_box_1<'a>(x: (BoxedTup<'a, '_>, &mut u32)) -> &'a mut u32 {
    &mut *(x.0).0
}

fn return_borrow_from_dropped_tupled_box_2<'a>(x: (BoxedTup<'a, '_>, &mut u32)) -> &'a mut u32 {
    (x.0).0
}

fn basic_tests() {
    let mut x = 2;
    let mut y = 3;
    let mut z = 4;
    *return_borrow_from_dropped_box_1(Box::new(&mut x)) += 10;
    assert_eq!((x, y, z), (12, 3, 4));
    *return_borrow_from_dropped_box_2(Box::new(&mut x)) += 10;
    assert_eq!((x, y, z), (22, 3, 4));
    *return_borrow_of_field_from_dropped_box_1(Box::new((&mut x, &mut y))) += 10;
    assert_eq!((x, y, z), (32, 3, 4));
    *return_borrow_of_field_from_dropped_box_2(Box::new((&mut x, &mut y))) += 10;
    assert_eq!((x, y, z), (42, 3, 4));
    *return_borrow_from_dropped_tupled_box_1((Box::new((&mut x, &mut y)), &mut z)) += 10;
    assert_eq!((x, y, z), (52, 3, 4));
    *return_borrow_from_dropped_tupled_box_2((Box::new((&mut x, &mut y)), &mut z)) += 10;
    assert_eq!((x, y, z), (62, 3, 4));
}

// These scribbling tests have been transcribed from
// issue-45696-scribble-on-boxed-borrow.rs
//
// In the context of that file, these tests are meant to show cases
// that should be *accepted* by the compiler, so here we are actually
// checking that the code we get when they are compiled matches our
// expectations.

struct Scribble<'a>(&'a mut u32);

impl<'a> Drop for Scribble<'a> { fn drop(&mut self) { *self.0 = 42; } }

// this is okay, in both AST-borrowck and NLL: The `Scribble` here *has*
// to strictly outlive `'a`
fn borrowed_scribble<'a>(s: &'a mut Scribble) -> &'a mut u32 {
    &mut *s.0
}

// this, by analogy to previous case, is also okay.
fn boxed_borrowed_scribble<'a>(s: Box<&'a mut Scribble>) -> &'a mut u32 {
    &mut *(*s).0
}

// this, by analogy to previous case, is also okay.
fn boxed_boxed_borrowed_scribble<'a>(s: Box<Box<&'a mut Scribble>>) -> &'a mut u32 {
    &mut *(**s).0
}

fn scribbling_tests() {
    let mut x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *borrowed_scribble(&mut long_lived) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
    x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_borrowed_scribble(Box::new(&mut long_lived)) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
    x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_boxed_borrowed_scribble(Box::new(Box::new(&mut long_lived))) += 10;
        assert_eq!(*long_lived.0, 11);
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    assert_eq!(x, 42);
}

fn main() {
    basic_tests();
    scribbling_tests();
}
