//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(improper_ctypes)]

// Crate use statements

#[cfg(FALSE)]
use flippity;

#[cfg(FALSE)]
static b: bool = false;

static b: bool = true;

mod rustrt {
    #[cfg(FALSE)]
    extern "C" {
        // This symbol doesn't exist and would be a link error if this
        // module was codegened
        pub fn FALSE();
    }

    extern "C" {}
}

#[cfg(FALSE)]
type t = isize;

type t = bool;

#[cfg(FALSE)]
enum tg {
    foo,
}

enum tg {
    bar,
}

#[cfg(FALSE)]
struct r {
    i: isize,
}

#[cfg(FALSE)]
fn r(i: isize) -> r {
    r { i: i }
}

struct r {
    i: isize,
}

fn r(i: isize) -> r {
    r { i: i }
}

#[cfg(FALSE)]
mod m {
    // This needs to parse but would fail in typeck. Since it's not in
    // the current config it should not be typechecked.
    pub fn FALSE() {
        return 0;
    }
}

mod m {
    // Submodules have slightly different code paths than the top-level
    // module, so let's make sure this jazz works here as well
    #[cfg(FALSE)]
    pub fn f() {}

    pub fn f() {}
}

// Since the FALSE configuration isn't defined main will just be
// parsed, but nothing further will be done with it
#[cfg(FALSE)]
pub fn main() {
    panic!()
}

pub fn main() {
    // Exercise some of the configured items in ways that wouldn't be possible
    // if they had the FALSE definition
    assert!((b));
    let _x: t = true;
    let _y: tg = tg::bar;

    test_in_fn_ctxt();
}

fn test_in_fn_ctxt() {
    #[cfg(FALSE)]
    fn f() {
        panic!()
    }
    fn f() {}
    f();

    #[cfg(FALSE)]
    static i: isize = 0;
    static i: isize = 1;
    assert_eq!(i, 1);
}

mod test_foreign_items {
    pub mod rustrt {
        extern "C" {
            #[cfg(FALSE)]
            pub fn write() -> String;
            pub fn write() -> String;
        }
    }
}

mod test_use_statements {
    #[cfg(FALSE)]
    use flippity_foo;
}

mod test_methods {
    struct Foo {
        bar: usize,
    }

    impl Fooable for Foo {
        #[cfg(FALSE)]
        fn what(&self) {}

        fn what(&self) {}

        #[cfg(FALSE)]
        fn the(&self) {}

        fn the(&self) {}
    }

    trait Fooable {
        #[cfg(FALSE)]
        fn what(&self);

        fn what(&self);

        #[cfg(FALSE)]
        fn the(&self);

        fn the(&self);
    }
}

#[cfg(any())]
mod nonexistent_file; // Check that unconfigured non-inline modules are not loaded or parsed.
