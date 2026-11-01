//@compile-flags: -Zmiri-strict-provenance

use std::thread;

fn create_and_detach() {
    thread::spawn(|| ());
}

fn create_and_join() {
    thread::spawn(|| ()).join().unwrap();
}

fn create_and_get_result() {
    let nine = thread::spawn(|| 5 + 4).join().unwrap();
    assert_eq!(nine, 9);
}

fn create_and_leak_result() {
    thread::spawn(|| 7);
}

fn create_nested_and_detach() {
    thread::spawn(|| {
        thread::spawn(|| ());
    });
}

fn create_nested_and_join() {
    let handle = thread::spawn(|| thread::spawn(|| ()));
    let handle_nested = handle.join().unwrap();
    handle_nested.join().unwrap();
}

fn create_move_in() {
    let x = String::from("Hello!");
    thread::spawn(move || {
        assert_eq!(x.len(), 6);
    })
    .join()
    .unwrap();
}

fn create_move_out() {
    let result = thread::spawn(|| String::from("Hello!")).join().unwrap();
    assert_eq!(result.len(), 6);
}

// This is not a data race!
fn shared_readonly() {
    use std::sync::Arc;

    let x = Arc::new(42i32);
    let h = thread::spawn({
        let x = Arc::clone(&x);
        move || {
            assert_eq!(*x, 42);
        }
    });

    assert_eq!(*x, 42);

    h.join().unwrap();
}

fn main() {
    create_and_detach();
    create_and_join();
    create_and_get_result();
    create_and_leak_result();
    create_nested_and_detach();
    create_nested_and_join();
    create_move_in();
    create_move_out();
    shared_readonly();
}
