//@ known-bug: #130425
//@ compile-flags: -Zmir-opt-level=5 -Zpolymorphize=on

struct S<T>(T)
where
    [T; (
        |_: u8| {
            static FOO: Sync = AtomicUsize::new(0);
            unsafe { &*(&FOO as *const _ as *const usize) }
        },
        1,
    )
        .1]: Copy;
