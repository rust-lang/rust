// All constant items (associated or otherwise) may syntactically use `_` as a name.

//@ check-pass

fn main() {}

#[cfg(FALSE)]
const _: () = {
    pub trait A {
        const _: () = ();
    }
    impl A for () {
        const _: () = ();
    }
    impl dyn A {
        const _: () = ();
    }
};
