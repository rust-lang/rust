// Check in two ways:
// - borrowck: Check with borrow checking errors when things are alive and dead.
// - runtime: Check with a mutable bool if things are dropped on time.
//
//@ revisions: runtime borrowck
//@ [runtime] run-pass
//@ [borrowck] check-fail

#![allow(dropping_references)]
#![feature(super_let, stmt_expr_attributes)]

use std::convert::identity;

struct DropMe<'a>(&'a mut bool);

impl Drop for DropMe<'_> {
    fn drop(&mut self) {
        *self.0 = true;
    }
}

// Check that a super let variable lives as long as the result of a block.
fn extended_variable() {
    let mut x = false;
    {
        let a = {
            super let b = DropMe(&mut x);
            &b
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true) // ok
}

// Check that the init expression of a super let is subject to (temporary) lifetime extension.
fn extended_temporary() {
    let mut x = false;
    {
        let a = {
            super let b = &DropMe(&mut x);
            b
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true); // ok
}

// Check that even non-extended temporaries live until the end of the block,
// but (unlike extended temporaries) not beyond that.
//
// This is necessary for things like select(pin!(identity(&temp()))) to work.
fn non_extended() {
    let mut x = false;
    {
        let _a = {
            // Use identity() to supress temporary lifetime extension.
            super let b = identity(&DropMe(&mut x));
            #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
            b
            // DropMe is still alive here...
        };
        // ... but not here.
        assert_eq!(x, true); // ok
    }
}

// Check that even non-extended temporaries live until the end of the block,
// but (unlike extended temporaries) not beyond that.
//
// This is necessary for things like select(pin!(identity(&temp()))) to work.
fn non_extended_in_expression() {
    let mut x = false;
    {
        identity((
            {
                // Use identity() to supress temporary lifetime extension.
                super let b = identity(&DropMe(&mut x));
                b
            },
            {
                #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
                // DropMe is still alive here...
            }
        ));
        // ... but not here.
        assert_eq!(x, true); // ok
    }
}

// Check `super let` in a match arm.
fn match_arm() {
    let mut x = false;
    {
        let a = match Some(123) {
            Some(_) => {
                super let b = DropMe(&mut x);
                &b
            }
            None => unreachable!(),
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true); // ok
}

// Check `super let` in an if body.
fn if_body() {
    let mut x = false;
    {
        let a = if true {
            super let b = DropMe(&mut x);
            &b
        } else {
            unreachable!()
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true); // ok
}

// Check `super let` in an else body.
fn else_body() {
    let mut x = false;
    {
        let a = if false {
            unreachable!()
        } else {
            super let b = DropMe(&mut x);
            &b
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true); // ok
}

fn without_initializer() {
    let mut x = false;
    {
        let a = {
            super let b;
            b = DropMe(&mut x);
            b
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true);
}

// Assignment isn't special, even when assigning to a `super let` variable.
fn assignment() {
    let mut x = false;
    {
        super let a;
        #[cfg(borrowck)] { a = &String::from("asdf"); }; //[borrowck]~ ERROR dropped while borrowed
        #[cfg(runtime)] { a = drop(&DropMe(&mut x)); } // Temporary dropped at the `;` as usual.
        assert_eq!(x, true);
        let _ = a;
    }
}

// `super let mut` should work just fine.
fn mutable() {
    let mut x = false;
    {
        let a = {
            super let mut b = None;
            &mut b
        };
        *a = Some(DropMe(&mut x));
    }
    assert_eq!(x, true);
}

// Temporary lifetime extension should recurse through `super let`s.
fn multiple_levels() {
    let mut x = false;
    {
        let a = {
            super let b = {
                super let c = {
                    super let d = &DropMe(&mut x);
                    d
                };
                c
            };
            b
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        drop(a);
        // DropMe is still alive here...
    }
    // ... but not here.
    assert_eq!(x, true);
}

// Non-extended temporaries should be dropped at the
// end of the first parent statement that isn't `super`.
fn multiple_levels_but_no_extension() {
    let mut x = false;
    {
        let _a = {
            super let b = {
                super let c = {
                    super let d = identity(&DropMe(&mut x));
                    d
                };
                c
            };
            #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
            b
            // DropMe is still alive here...
        };
        // ... but not here.
        assert_eq!(x, true);
    }
}

// Check for potential weird interactions with `let else`.
fn super_let_and_let_else() {
    let mut x = false;
    {
        let a = 'a: {
            let Some(_) = Some(123) else { unreachable!() };
            super let b = DropMe(&mut x);
            let None = Some(123) else { break 'a &b };
            unreachable!()
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        // DropMe is still alive here...
        drop(a);
    }
    // ... but not here.
    assert_eq!(x, true);
}

// Check if `super let .. else ..;` works.
fn super_let_else() {
    let mut x = false;
    {
        let a = {
            let dropme = Some(DropMe(&mut x));
            super let Some(x) = dropme else { unreachable!() };
            &x
        };
        #[cfg(borrowck)] { x = true; } //[borrowck]~ ERROR borrowed
        // DropMe is still alive here...
        drop(a);
    }
    // ... but not here.
    assert_eq!(x, true);
}

fn main() {
    extended_variable();
    extended_temporary();
    non_extended();
    non_extended_in_expression();
    match_arm();
    if_body();
    else_body();
    without_initializer();
    assignment();
    mutable();
    multiple_levels();
    multiple_levels_but_no_extension();
    super_let_and_let_else();
    super_let_else();
}
