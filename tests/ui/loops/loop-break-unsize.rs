// Regression test for #62312
//@ check-pass

fn main() {
    let _ = loop {
        break Box::new(()) as Box<dyn Send>;
    };
}
