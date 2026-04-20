// Regression test for https://github.com/rust-lang/rust/issues/139089

fn foo(x: ＆Vec<u8>) {
    //~^ ERROR unknown start of token
    x.push(0);
    //~^ ERROR cannot borrow `*x` as mutable
}

fn main() {}
