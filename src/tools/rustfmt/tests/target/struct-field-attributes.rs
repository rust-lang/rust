// #1535
#![feature(struct_field_attributes)]

struct Foo {
    bar: u64,

    #[cfg(test)]
    qux: u64,
}

fn do_something() -> Foo {
    Foo {
        bar: 0,

        #[cfg(test)]
        qux: 1,
    }
}

fn main() {
    do_something();
}

// #1462
struct Foo {
    foo: usize,
    #[cfg(feature = "include-bar")]
    bar: usize,
}

fn new_foo() -> Foo {
    Foo {
        foo: 0,
        #[cfg(feature = "include-bar")]
        bar: 0,
    }
}

// #2044
pub enum State {
    Closure(
        #[cfg_attr(
            feature = "serde_derive",
            serde(state_with = "::serialization::closure")
        )]
        GcPtr<ClosureData>,
    ),
}

struct Fields(
    #[cfg_attr(
        feature = "serde_derive",
        serde(state_with = "::base::serialization::shared")
    )]
    Arc<Vec<InternedStr>>,
);

// #2309
pub struct A {
    #[doc = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"]
    pub foos: Vec<bool>,
}
