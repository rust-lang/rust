// Ensure that thread_local init with `const { 0 }` still has unique address at run-time
#[test]
fn waker_current_thread_id() {
    let first = super::waker::current_thread_id();
    let t = crate::thread::spawn(move || {
        let second = super::waker::current_thread_id();
        assert_ne!(first, second);
        assert_eq!(second, super::waker::current_thread_id());
    });

    assert_eq!(first, super::waker::current_thread_id());
    t.join().unwrap();
    assert_eq!(first, super::waker::current_thread_id());
}
