//! Test that destructors for temporaries run either at end of
//! statement or end of block as appropriate.

//@ run-pass

#![feature(box_patterns)]

static mut FLAGS: u64 = 0;

struct Box<T> {
    f: T,
}

struct AddFlags {
    bits: u64,
}

fn add_flags(bits: u64) -> AddFlags {
    AddFlags { bits }
}

fn arg(expected: u64, _x: &AddFlags) {
    check_flags(expected);
}

fn pass<T>(v: T) -> T {
    v
}

fn check_flags(expected: u64) {
    unsafe {
        let actual = FLAGS;
        FLAGS = 0;
        assert_eq!(actual, expected, "flags {}, expected {}", actual, expected);
    }
}

impl AddFlags {
    fn check_flags(&self, expected: u64) -> &AddFlags {
        check_flags(expected);
        self
    }

    fn bits(&self) -> u64 {
        self.bits
    }
}

impl Drop for AddFlags {
    fn drop(&mut self) {
        unsafe {
            FLAGS += self.bits;
        }
    }
}

macro_rules! end_of_block {
    ($pat:pat, $expr:expr) => {{
        {
            let $pat = $expr;
            check_flags(0);
        }
        check_flags(1);
    }};
}

macro_rules! end_of_stmt {
    ($pat:pat, $expr:expr) => {{
        {
            let $pat = $expr;
            check_flags(1);
        }
        check_flags(0);
    }};
}

fn main() {
    end_of_block!(_x, add_flags(1));
    end_of_block!(_x, &add_flags(1));
    end_of_block!(_x, &&add_flags(1));
    end_of_block!(_x, Box { f: add_flags(1) });
    end_of_block!(_x, Box { f: &add_flags(1) });
    end_of_block!(_x, pass(add_flags(1)));
    end_of_block!(ref _x, add_flags(1));
    end_of_block!(AddFlags { bits: ref _x }, add_flags(1));
    end_of_block!(&AddFlags { bits: _ }, &add_flags(1));
    end_of_block!((_, ref _y), (add_flags(1), 22));
    end_of_block!(box ref _x, std::boxed::Box::new(add_flags(1)));
    end_of_block!(box _x, std::boxed::Box::new(add_flags(1)));
    end_of_block!(_, {
        {
            check_flags(0);
            &add_flags(1)
        }
    });
    end_of_block!(_, &((Box { f: add_flags(1) }).f));
    end_of_block!(_, &(([add_flags(1)])[0]));

    end_of_stmt!(_, add_flags(1));
    end_of_stmt!((_, _), (add_flags(1), 22));
    end_of_stmt!(ref _x, arg(0, &add_flags(1)));
    end_of_stmt!(ref _x, add_flags(1).check_flags(0).bits());
    end_of_stmt!(AddFlags { bits: _ }, add_flags(1));
}
