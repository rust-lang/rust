#![allow(non_local_definitions, clippy::needless_if)]
#![warn(clippy::bool_comparison)]
#![allow(clippy::non_canonical_partial_ord_impl, clippy::nonminimal_bool)]

fn main() {
    let x = true;
    if x == true {
        "yes"
    } else {
        "no"
    };
    if x == false {
        "yes"
    } else {
        "no"
    };
    if true == x {
        "yes"
    } else {
        "no"
    };
    if false == x {
        "yes"
    } else {
        "no"
    };
    if x != true {
        "yes"
    } else {
        "no"
    };
    if x != false {
        "yes"
    } else {
        "no"
    };
    if true != x {
        "yes"
    } else {
        "no"
    };
    if false != x {
        "yes"
    } else {
        "no"
    };
    if x < true {
        "yes"
    } else {
        "no"
    };
    if false < x {
        "yes"
    } else {
        "no"
    };
    if x > false {
        "yes"
    } else {
        "no"
    };
    if true > x {
        "yes"
    } else {
        "no"
    };
    let y = true;
    if x < y {
        "yes"
    } else {
        "no"
    };
    if x > y {
        "yes"
    } else {
        "no"
    };
}

#[allow(dead_code)]
fn issue3703() {
    struct Foo;
    impl PartialEq<bool> for Foo {
        fn eq(&self, _: &bool) -> bool {
            true
        }
    }
    impl PartialEq<Foo> for bool {
        fn eq(&self, _: &Foo) -> bool {
            true
        }
    }
    impl PartialOrd<bool> for Foo {
        fn partial_cmp(&self, _: &bool) -> Option<std::cmp::Ordering> {
            None
        }
    }
    impl PartialOrd<Foo> for bool {
        fn partial_cmp(&self, _: &Foo) -> Option<std::cmp::Ordering> {
            None
        }
    }

    if Foo == true {}
    if true == Foo {}
    if Foo != true {}
    if true != Foo {}
    if Foo == false {}
    if false == Foo {}
    if Foo != false {}
    if false != Foo {}
    if Foo < false {}
    if false < Foo {}
}

#[allow(dead_code)]
fn issue4983() {
    let a = true;
    let b = false;

    if a == !b {};
    if !a == b {};
    if a == b {};
    if !a == !b {};

    if b == !a {};
    if !b == a {};
    if b == a {};
    if !b == !a {};
}

macro_rules! m {
    ($func:ident) => {
        $func()
    };
}

fn func() -> bool {
    true
}

#[allow(dead_code)]
fn issue3973() {
    // ok, don't lint on `cfg` invocation
    if false == cfg!(feature = "debugging") {}
    if cfg!(feature = "debugging") == false {}
    if true == cfg!(feature = "debugging") {}
    if cfg!(feature = "debugging") == true {}

    // lint, could be simplified
    if false == m!(func) {}
    if m!(func) == false {}
    if true == m!(func) {}
    if m!(func) == true {}

    // no lint with a variable
    let is_debug = false;
    if is_debug == cfg!(feature = "debugging") {}
    if cfg!(feature = "debugging") == is_debug {}
    if is_debug == m!(func) {}
    if m!(func) == is_debug {}
    let is_debug = true;
    if is_debug == cfg!(feature = "debugging") {}
    if cfg!(feature = "debugging") == is_debug {}
    if is_debug == m!(func) {}
    if m!(func) == is_debug {}
}

#[allow(clippy::unnecessary_cast)]
fn issue9907() {
    let _ = ((1 < 2) == false) as usize;
    let _ = (false == m!(func)) as usize;
    // This is not part of the issue, but an unexpected found when fixing the issue,
    // the provided span was inside of macro rather than the macro callsite.
    let _ = ((1 < 2) == !m!(func)) as usize;
}
