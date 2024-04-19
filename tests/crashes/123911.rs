//@ known-bug: #123911

macro_rules! m {
    ($attr_path: path) => {
        #[$attr_path]
        fn f() {}
    }
}

m!(inline<{
    let a = CharCharFloat { a: 1 };
    let b = rustrt::rust_dbg_abi_4(a);
    println!("a: {}", b.a);
}>);

fn main() {}
