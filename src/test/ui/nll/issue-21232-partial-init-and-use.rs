// This test enumerates various cases of interest for partial
// [re]initialization of ADTs and tuples.
//
// See rust-lang/rust#21232, rust-lang/rust#54986, and rust-lang/rust#54987.
//
// All of tests in this file are expected to change from being
// rejected, at least under NLL (by rust-lang/rust#54986) to being
// **accepted** when rust-lang/rust#54987 is implemented.
// (That's why there are assertions in the code.)
//
// See issue-21232-partial-init-and-erroneous-use.rs for cases of
// tests that are meant to continue failing to compile once
// rust-lang/rust#54987 is implemented.

#![feature(nll)]

struct S<Y> {
    x: u32,

    // Note that even though `y` may implement `Drop`, under #54987 we
    // will still allow partial initialization of `S` itself.
    y: Y,
}

enum Void { }

type B = Box<u32>;

impl S<B> { fn new() -> Self { S { x: 0, y: Box::new(0) } } }

fn borrow_s(s: &S<B>) { assert_eq!(s.x, 10); assert_eq!(*s.y, 20); }
fn move_s(s: S<B>) {  assert_eq!(s.x, 10); assert_eq!(*s.y, 20); }
fn borrow_field(x: &u32) { assert_eq!(*x, 10); }

type T = (u32, B);
type Tvoid = (u32, Void);

fn borrow_t(t: &T) { assert_eq!(t.0, 10); assert_eq!(*t.1, 20); }
fn move_t(t: T) {  assert_eq!(t.0, 10); assert_eq!(*t.1, 20); }

struct Q<F> {
    v: u32,
    r: R<F>,
}

struct R<F> {
    w: u32,
    f: F,
}

impl<F> Q<F> { fn new(f: F) -> Self { Q { v: 0, r: R::new(f) } } }
impl<F> R<F> { fn new(f: F) -> Self { R { w: 0, f } } }

// Axes to cover:
// * local/field: Is the structure in a local or a field
// * fully/partial/void: Are we fully initializing it before using any part?
//                       Is whole type empty due to a void component?
// * init/reinit: First initialization, or did we previously inititalize and then move out?
// * struct/tuple: Is this a struct or a (X, Y).
//
// As a shorthand for the cases above, adding a numeric summary to
// each test's fn name to denote each point on each axis.
//
// e.g., 1000 = field fully init struct; 0211 = local void reinit tuple

// It got pretty monotonous writing the same code over and over, and I
// feared I would forget details. So I abstracted some desiderata into
// macros. But I left the initialization code inline, because that's
// where the errors for #54986 will be emitted.

macro_rules! use_fully {
    (struct $s:expr) => { {
        borrow_field(& $s.x );
        borrow_s(& $s );
        move_s( $s );
    } };

    (tuple $t:expr) => { {
        borrow_field(& $t.0 );
        borrow_t(& $t );
        move_t( $t );
    } }
}

macro_rules! use_part {
    (struct $s:expr) => { {
        borrow_field(& $s.x );
        match $s { S { ref x, y: _ } => { borrow_field(x); } }
    } };

    (tuple $t:expr) => { {
        borrow_field(& $t.0 );
        match $t { (ref x, _) => { borrow_field(x); } }
    } }
}

fn test_0000_local_fully_init_and_use_struct() {
    let s: S<B>;
    s.x = 10; s.y = Box::new(20);
    //~^ ERROR assign to part of possibly uninitialized variable: `s` [E0381]
    use_fully!(struct s);
}

fn test_0001_local_fully_init_and_use_tuple() {
    let t: T;
    t.0 = 10; t.1 = Box::new(20);
    //~^ ERROR assign to part of possibly uninitialized variable: `t` [E0381]
    use_fully!(tuple t);
}

fn test_0010_local_fully_reinit_and_use_struct() {
    let mut s: S<B> = S::new(); drop(s);
    s.x = 10; s.y = Box::new(20);
    //~^ ERROR assign to part of moved value: `s` [E0382]
    use_fully!(struct s);
}

fn test_0011_local_fully_reinit_and_use_tuple() {
    let mut t: T = (0, Box::new(0)); drop(t);
    t.0 = 10; t.1 = Box::new(20);
    //~^ ERROR assign to part of moved value: `t` [E0382]
    use_fully!(tuple t);
}

fn test_0100_local_partial_init_and_use_struct() {
    let s: S<B>;
    s.x = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `s` [E0381]
    use_part!(struct s);
}

fn test_0101_local_partial_init_and_use_tuple() {
    let t: T;
    t.0 = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `t` [E0381]
    use_part!(tuple t);
}

fn test_0110_local_partial_reinit_and_use_struct() {
    let mut s: S<B> = S::new(); drop(s);
    s.x = 10;
    //~^ ERROR assign to part of moved value: `s` [E0382]
    use_part!(struct s);
}

fn test_0111_local_partial_reinit_and_use_tuple() {
    let mut t: T = (0, Box::new(0)); drop(t);
    t.0 = 10;
    //~^ ERROR assign to part of moved value: `t` [E0382]
    use_part!(tuple t);
}

fn test_0200_local_void_init_and_use_struct() {
    let s: S<Void>;
    s.x = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `s` [E0381]
    use_part!(struct s);
}

fn test_0201_local_void_init_and_use_tuple() {
    let t: Tvoid;
    t.0 = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `t` [E0381]
    use_part!(tuple t);
}

// NOTE: uniform structure of tests here makes n21n (aka combining
// Void with Reinit) an (even more) senseless case, as we cannot
// safely create initial instance containing Void to move out of and
// then reinitialize. While I was tempted to sidestep this via some
// unsafe code (eek), lets just instead not encode such tests.

// fn test_0210_local_void_reinit_and_use_struct() { unimplemented!() }
// fn test_0211_local_void_reinit_and_use_tuple() { unimplemented!() }

fn test_1000_field_fully_init_and_use_struct() {
    let q: Q<S<B>>;
    q.r.f.x = 10; q.r.f.y = Box::new(20);
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_fully!(struct q.r.f);
}

fn test_1001_field_fully_init_and_use_tuple() {
    let q: Q<T>;
    q.r.f.0 = 10; q.r.f.1 = Box::new(20);
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_fully!(tuple q.r.f);
}

fn test_1010_field_fully_reinit_and_use_struct() {
    let mut q: Q<S<B>> = Q::new(S::new()); drop(q.r);
    q.r.f.x = 10; q.r.f.y = Box::new(20);
    //~^ ERROR assign to part of moved value: `q.r` [E0382]
    use_fully!(struct q.r.f);
}

fn test_1011_field_fully_reinit_and_use_tuple() {
    let mut q: Q<T> = Q::new((0, Box::new(0))); drop(q.r);
    q.r.f.0 = 10; q.r.f.1 = Box::new(20);
    //~^ ERROR assign to part of moved value: `q.r` [E0382]
    use_fully!(tuple q.r.f);
}

fn test_1100_field_partial_init_and_use_struct() {
    let q: Q<S<B>>;
    q.r.f.x = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_part!(struct q.r.f);
}

fn test_1101_field_partial_init_and_use_tuple() {
    let q: Q<T>;
    q.r.f.0 = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_part!(tuple q.r.f);
}

fn test_1110_field_partial_reinit_and_use_struct() {
    let mut q: Q<S<B>> = Q::new(S::new()); drop(q.r);
    q.r.f.x = 10;
    //~^ ERROR assign to part of moved value: `q.r` [E0382]
    use_part!(struct q.r.f);
}

fn test_1111_field_partial_reinit_and_use_tuple() {
    let mut q: Q<T> = Q::new((0, Box::new(0))); drop(q.r);
    q.r.f.0 = 10;
    //~^ ERROR assign to part of moved value: `q.r` [E0382]
    use_part!(tuple q.r.f);
}

fn test_1200_field_void_init_and_use_struct() {
    let mut q: Q<S<Void>>;
    q.r.f.x = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_part!(struct q.r.f);
}

fn test_1201_field_void_init_and_use_tuple() {
    let mut q: Q<Tvoid>;
    q.r.f.0 = 10;
    //~^ ERROR assign to part of possibly uninitialized variable: `q` [E0381]
    use_part!(tuple q.r.f);
}

// See NOTE abve.

// fn test_1210_field_void_reinit_and_use_struct() { unimplemented!() }
// fn test_1211_field_void_reinit_and_use_tuple() { unimplemented!() }

// The below are some additional cases of interest that have been
// transcribed from other bugs based on old erroneous codegen when we
// encountered partial writes.

fn issue_26996() {
    let mut c = (1, "".to_owned());
    match c {
        c2 => {
            c.0 = 2; //~ ERROR assign to part of moved value
            assert_eq!(c2.0, 1);
        }
    }
}

fn issue_27021() {
    let mut c = (1, (1, "".to_owned()));
    match c {
        c2 => {
            (c.1).0 = 2; //~ ERROR assign to part of moved value
            assert_eq!((c2.1).0, 1);
        }
    }

    let mut c = (1, (1, (1, "".to_owned())));
    match c.1 {
        c2 => {
            ((c.1).1).0 = 3; //~ ERROR assign to part of moved value
            assert_eq!((c2.1).0, 1);
        }
    }
}

fn main() {
    test_0000_local_fully_init_and_use_struct();
    test_0001_local_fully_init_and_use_tuple();
    test_0010_local_fully_reinit_and_use_struct();
    test_0011_local_fully_reinit_and_use_tuple();
    test_0100_local_partial_init_and_use_struct();
    test_0101_local_partial_init_and_use_tuple();
    test_0110_local_partial_reinit_and_use_struct();
    test_0111_local_partial_reinit_and_use_tuple();
    test_0200_local_void_init_and_use_struct();
    test_0201_local_void_init_and_use_tuple();
    // test_0210_local_void_reinit_and_use_struct();
    // test_0211_local_void_reinit_and_use_tuple();
    test_1000_field_fully_init_and_use_struct();
    test_1001_field_fully_init_and_use_tuple();
    test_1010_field_fully_reinit_and_use_struct();
    test_1011_field_fully_reinit_and_use_tuple();
    test_1100_field_partial_init_and_use_struct();
    test_1101_field_partial_init_and_use_tuple();
    test_1110_field_partial_reinit_and_use_struct();
    test_1111_field_partial_reinit_and_use_tuple();
    test_1200_field_void_init_and_use_struct();
    test_1201_field_void_init_and_use_tuple();
    // test_1210_field_void_reinit_and_use_struct();
    // test_1211_field_void_reinit_and_use_tuple();

    issue_26996();
    issue_27021();
}
