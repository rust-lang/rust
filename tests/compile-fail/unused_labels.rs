#![plugin(clippy)]
#![feature(plugin)]

#![allow(dead_code, items_after_statements)]
#![deny(unused_label)]

fn unused_label() {
    'label: for i in 1..2 { //~ERROR: unused label `'label`
        if i > 4 { continue }
    }
}

fn foo() {
    'same_label_in_two_fns: loop {
        break 'same_label_in_two_fns;
    }
}


fn bla() {
    'a: loop { break } //~ERROR: unused label `'a`
    fn blub() {}
}

fn main() {
    'a: for _ in 0..10 {
        while let Some(42) = None {
            continue 'a;
        }
    }

    'same_label_in_two_fns: loop { //~ERROR: unused label `'same_label_in_two_fns`
        let _ = 1;
    }
}
