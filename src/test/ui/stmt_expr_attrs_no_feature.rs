#![feature(custom_attribute)]

macro_rules! stmt_mac {
    () => {
        fn b() {}
    }
}

fn main() {
    #[attr]
    fn a() {}

    #[attr] //~ ERROR attributes on expressions are experimental
    {

    }

    #[attr]
    5;

    #[attr]
    stmt_mac!();
}

// Check that cfg works right

#[cfg(unset)]
fn c() {
    #[attr]
    5;
}

#[cfg(not(unset))]
fn j() {
    #[attr]
    5;
}

#[cfg_attr(not(unset), cfg(unset))]
fn d() {
    #[attr]
    8;
}

#[cfg_attr(not(unset), cfg(not(unset)))]
fn i() {
    #[attr]
    8;
}

// check that macro expansion and cfg works right

macro_rules! item_mac {
    ($e:ident) => {
        fn $e() {
            #[attr]
            42;

            #[cfg(unset)]
            fn f() {
                #[attr]
                5;
            }

            #[cfg(not(unset))]
            fn k() {
                #[attr]
                5;
            }

            #[cfg_attr(not(unset), cfg(unset))]
            fn g() {
                #[attr]
                8;
            }

            #[cfg_attr(not(unset), cfg(not(unset)))]
            fn h() {
                #[attr]
                8;
            }

        }
    }
}

item_mac!(e);

// check that the gate visitor works right:

extern {
    #[cfg(unset)]
    fn x(a: [u8; #[attr] 5]);
    fn y(a: [u8; #[attr] 5]); //~ ERROR attributes on expressions are experimental
}

struct Foo;
impl Foo {
    #[cfg(unset)]
    const X: u8 = #[attr] 5;
    const Y: u8 = #[attr] 5; //~ ERROR attributes on expressions are experimental
}

trait Bar {
    #[cfg(unset)]
    const X: [u8; #[attr] 5];
    const Y: [u8; #[attr] 5]; //~ ERROR attributes on expressions are experimental
}

struct Joyce {
    #[cfg(unset)]
    field: [u8; #[attr] 5],
    field2: [u8; #[attr] 5] //~ ERROR attributes on expressions are experimental
}

struct Walky(
    #[cfg(unset)] [u8; #[attr] 5],
    [u8; #[attr] 5] //~ ERROR attributes on expressions are experimental
);

enum Mike {
    Happy(
        #[cfg(unset)] [u8; #[attr] 5],
        [u8; #[attr] 5] //~ ERROR attributes on expressions are experimental
    ),
    Angry {
        #[cfg(unset)]
        field: [u8; #[attr] 5],
        field2: [u8; #[attr] 5] //~ ERROR attributes on expressions are experimental
    }
}

fn pat() {
    match 5 {
        #[cfg(unset)]
        5 => #[attr] (),
        6 => #[attr] (), //~ ERROR attributes on expressions are experimental
        _ => (),
    }
}
