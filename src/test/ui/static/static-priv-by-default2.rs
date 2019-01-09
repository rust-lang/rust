// aux-build:static_priv_by_default.rs

extern crate static_priv_by_default;

mod child {
    pub mod childs_child {
        static private: isize = 0;
        pub static public: isize = 0;
    }
}

fn foo<T>(_: T) {}

fn test1() {
    use child::childs_child::private;
    //~^ ERROR: static `private` is private
    use child::childs_child::public;

    foo(private);
}

fn test2() {
    use static_priv_by_default::private;
    //~^ ERROR: static `private` is private
    use static_priv_by_default::public;

    foo(private);
}

fn main() {}
