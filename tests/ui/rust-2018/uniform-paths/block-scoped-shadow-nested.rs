// edition:2018

mod my {
    pub mod sub {
        pub fn bar() {}
    }
}

mod sub {
    pub fn bar() {}
}

fn foo() {
    use my::sub;
    {
        use sub::bar; //~ ERROR `sub` is ambiguous
    }
}

fn main() {}
