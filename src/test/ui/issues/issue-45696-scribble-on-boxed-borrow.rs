// rust-lang/rust#45696: This test is checking that we *cannot* return
// mutable borrows that would be scribbled over by destructors before
// the return occurs.
//
// We will explicitly test NLL, and migration modes;
// thus we will also skip the automated compare-mode=nll.

// revisions: nll migrate
// ignore-compare-mode-nll

// This test is going to pass in the migrate revision, because the AST-borrowck
// accepted this code in the past (see notes below). So we use `#[rustc_error]`
// to keep the outcome as an error in all scenarios, and rely on the stderr
// files to show what the actual behavior is. (See rust-lang/rust#49855.)
#![feature(rustc_attrs)]

#![cfg_attr(nll, feature(nll))]

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

// this is not okay: in between the time that we take the mutable
// borrow and the caller receives it as a return value, the drop of
// `s` will scribble on it, violating our aliasing guarantees.
//
// * (Maybe in the future the two-phase borrows system will be
//   extended to support this case. But for now, it is an error in
//   NLL, even with two-phase borrows.)
//
// In any case, the AST-borrowck was not smart enough to know that
// this should be an error. (Which is perhaps the essence of why
// rust-lang/rust#45696 arose in the first place.)
fn scribbled<'a>(s: Scribble<'a>) -> &'a mut u32 {
    &mut *s.0 //[nll]~ ERROR borrow may still be in use when destructor runs [E0713]
    //[migrate]~^ WARNING borrow may still be in use when destructor runs [E0713]
    //[migrate]~| WARNING this error has been downgraded to a warning for backwards compatibility
    //[migrate]~| WARNING this represents potential undefined behavior in your code
}

// This, by analogy to previous case, is *also* not okay.
//
// (But again, AST-borrowck was not smart enogh to know that this
// should be an error.)
fn boxed_scribbled<'a>(s: Box<Scribble<'a>>) -> &'a mut u32 {
    &mut *(*s).0 //[nll]~ ERROR borrow may still be in use when destructor runs [E0713]
    //[migrate]~^ WARNING borrow may still be in use when destructor runs [E0713]
    //[migrate]~| WARNING this error has been downgraded to a warning for backwards compatibility
    //[migrate]~| WARNING this represents potential undefined behavior in your code
}

// This, by analogy to previous case, is *also* not okay.
//
// (But again, AST-borrowck was not smart enogh to know that this
// should be an error.)
fn boxed_boxed_scribbled<'a>(s: Box<Box<Scribble<'a>>>) -> &'a mut u32 {
    &mut *(**s).0 //[nll]~ ERROR borrow may still be in use when destructor runs [E0713]
    //[migrate]~^ WARNING borrow may still be in use when destructor runs [E0713]
    //[migrate]~| WARNING this error has been downgraded to a warning for backwards compatibility
    //[migrate]~| WARNING this represents potential undefined behavior in your code
}

#[rustc_error]
fn main() { //[migrate]~ ERROR compilation successful
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
