#[test]
fn waker_current_thread_id() {
    let first = super::waker::current_thread_id();
    let t = crate::thread::spawn(|| {
        let second = super::waker::current_thread_id();
        assert_ne!(first, second);
        assert_eq!(second, super::waker::current_thread_id());
    });

    assert_eq!(first, super::waker::current_thread_id());
    t.join().unwrap();
    assert_eq!(first, super::waker::current_thread_id());
}
