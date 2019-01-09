// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

struct Element;

macro_rules! foo {
    ($tag: expr, $string: expr) => {
        if $tag == $string {
            let element: Box<_> = box Element;
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
