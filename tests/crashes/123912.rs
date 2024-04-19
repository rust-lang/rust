//@ known-bug: #123912

macro_rules! m {
    ($attr_path: path) => {
        #[$attr_path]
        fn f() {}
    }
}

m!(inline<{
    let a = CharCharFloat { a: 1 };
    println!("a: {}", a);
}>);

fn main() {}
