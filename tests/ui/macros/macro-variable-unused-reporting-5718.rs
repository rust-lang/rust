// https://github.com/rust-lang/rust/issues/5718
//@ run-pass

struct Element;

macro_rules! foo {
    ($tag: expr, $string: expr) => {
        if $tag == $string {
            let element: Box<_> = Box::new(Element);
            unsafe {
                return std::mem::transmute::<_, usize>(element);
            }
        }
    }
}

fn bar() -> usize {
    foo!("a", "b");
    0
}

fn main() {
    bar();
}
