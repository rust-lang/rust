//@ compile-flags: -Zunpretty=hir

// issue#97006

macro_rules! m {
    ($attr_path: path) => {
        #[$attr_path]
        fn f() {}
    }
}

m!(inline<u8>); //~ ERROR: unexpected generic arguments in path

fn main() {}
