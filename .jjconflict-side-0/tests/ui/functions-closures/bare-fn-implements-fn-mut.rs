//! Tests that bare functions implement the `FnMut` trait.
//!
//! See <https://github.com/rust-lang/rust/issues/15448>.

//@ run-pass

fn call_f<F:FnMut()>(mut f: F) {
    f();
}

fn f() {
    println!("hello");
}

fn call_g<G:FnMut(String,String) -> String>(mut g: G, x: String, y: String)
          -> String {
    g(x, y)
}

fn g(mut x: String, y: String) -> String {
    x.push_str(&y);
    x
}

fn main() {
    call_f(f);
    assert_eq!(call_g(g, "foo".to_string(), "bar".to_string()),
               "foobar");
}
