use std::sync::Arc;
use std::rc::Rc;

fn foo1(_: Arc<usize>) {}
fn bar1(_: Arc<usize>) {}
fn test_arc() {
    let x = Arc::new(1);
    foo1(x);
    foo1(x); //~ ERROR use of moved value
    bar1(x); //~ ERROR use of moved value
}

fn foo2(_: Rc<usize>) {}
fn bar2(_: Rc<usize>) {}
fn test_rc() {
    let x = Rc::new(1);
    foo2(x);
    foo2(x); //~ ERROR use of moved value
    bar2(x); //~ ERROR use of moved value
}

fn test_closure() {
    let x = Arc::new(1);
    for _ in 0..4 {
        // Ideally we should suggest `let x = x.clone();` here.
        std::thread::spawn(move || { //~ ERROR use of moved value
            println!("{}", x);
        });
    }
}

fn main() {
    test_rc();
    test_arc();
    test_closure();
}
