// rust-lang/rust#45696: This test is checking that we *cannot* return
// mutable borrows that would be scribbled over by destructors before
// the return occurs.

//@ ignore-compare-mode-polonius

struct Scribble<'a>(&'a mut u32);

impl<'a> Drop for Scribble<'a> { fn drop(&mut self) { *self.0 = 42; } }

// this is okay: The `Scribble` here *has* to strictly outlive `'a`
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

// this is not okay: in between the time that we take the mutable
// borrow and the caller receives it as a return value, the drop of
// `s` will scribble on it, violating our aliasing guarantees.
//
// * (Maybe in the future the two-phase borrows system will be
//   extended to support this case. But for now, it is an error in
//   NLL, even with two-phase borrows.)
fn scribbled<'a>(s: Scribble<'a>) -> &'a mut u32 {
    &mut *s.0 //~ ERROR borrow may still be in use when destructor runs [E0713]
}

// This, by analogy to previous case, is *also* not okay.
fn boxed_scribbled<'a>(s: Box<Scribble<'a>>) -> &'a mut u32 {
    &mut *(*s).0 //~ ERROR borrow may still be in use when destructor runs [E0713]
}

// This, by analogy to previous case, is *also* not okay.
fn boxed_boxed_scribbled<'a>(s: Box<Box<Scribble<'a>>>) -> &'a mut u32 {
    &mut *(**s).0 //~ ERROR borrow may still be in use when destructor runs [E0713]
}

fn main() {
    let mut x = 1;
    {
        let mut long_lived = Scribble(&mut x);
        *borrowed_scribble(&mut long_lived) += 10;
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_borrowed_scribble(Box::new(&mut long_lived)) += 10;
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    {
        let mut long_lived = Scribble(&mut x);
        *boxed_boxed_borrowed_scribble(Box::new(Box::new(&mut long_lived))) += 10;
        // (Scribble dtor runs here, after `&mut`-borrow above ends)
    }
    *scribbled(Scribble(&mut x)) += 10;
    *boxed_scribbled(Box::new(Scribble(&mut x))) += 10;
    *boxed_boxed_scribbled(Box::new(Box::new(Scribble(&mut x)))) += 10;
}
