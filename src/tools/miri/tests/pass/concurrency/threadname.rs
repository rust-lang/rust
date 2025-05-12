use std::thread;

fn main() {
    // When we have not set the name...
    thread::spawn(|| {
        assert!(thread::current().name().is_none());
    });

    // ... and when we have set it.
    thread::Builder::new()
        .name("childthread".to_string())
        .spawn(move || {
            assert_eq!(thread::current().name().unwrap(), "childthread");
        })
        .unwrap()
        .join()
        .unwrap();

    // Long thread name.
    let long_name = std::iter::once("test_named_thread_truncation")
        .chain(std::iter::repeat(" long").take(100))
        .collect::<String>();
    thread::Builder::new()
        .name(long_name.clone())
        .spawn(move || {
            assert_eq!(thread::current().name().unwrap(), long_name);
        })
        .unwrap()
        .join()
        .unwrap();

    // Also check main thread name.
    assert_eq!(thread::current().name().unwrap(), "main");
}
