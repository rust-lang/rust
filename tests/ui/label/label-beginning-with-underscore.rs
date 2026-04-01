//@ check-pass

#![deny(unused_labels)]

fn main() {
    // `unused_label` shouldn't warn labels beginning with `_`
    '_unused: loop {
        break;
    }
}
