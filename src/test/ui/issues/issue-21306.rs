// run-pass

use std::sync::Arc;

fn main() {
    let x = 5;
    let command = Arc::new(Box::new(|| { x*2 }));
    assert_eq!(command(), 10);
}
