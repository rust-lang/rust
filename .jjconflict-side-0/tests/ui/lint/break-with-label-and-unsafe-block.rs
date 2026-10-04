//@ check-pass

#![deny(break_with_label_and_loop)]

unsafe fn foo() -> i32 { 42 }

fn main () {
    'label: loop {
        break 'label unsafe { foo() }
    };
}
