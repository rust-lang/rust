use std::sync::Mutex;
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

    // Thread name during TLS dtor.
    during_dtor();
}

fn during_dtor() {
    // Go through one round of TLS destruction first.
    struct Defer;
    impl Drop for Defer {
        fn drop(&mut self) {
            RETRIEVE.with(|_| {});
        }
    }

    struct RetrieveName;
    impl Drop for RetrieveName {
        fn drop(&mut self) {
            *NAME.lock().unwrap() = Some(thread::current().name().unwrap().to_owned());
        }
    }

    static NAME: Mutex<Option<String>> = Mutex::new(None);

    thread_local! {
        static DEFER: Defer = const { Defer };
        static RETRIEVE: RetrieveName = const { RetrieveName };
    }

    let t = thread::Builder::new().name("test".to_owned()).spawn(|| DEFER.with(|_| {})).unwrap();
    t.join().unwrap();

    let name = NAME.lock().unwrap();
    let name = name.as_ref().unwrap();
    assert_eq!(name, "test");
}
