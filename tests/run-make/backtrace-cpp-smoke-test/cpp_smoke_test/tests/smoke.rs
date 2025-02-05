use std::sync::atomic::{AtomicBool, Ordering};

extern "C" {
    fn cpp_trampoline(func: extern "C" fn()) -> ();
}

#[test]
fn smoke_test_cpp() {
    static RAN_ASSERTS: AtomicBool = AtomicBool::new(false);

    extern "C" fn assert_cpp_frames() {
        let mut physical_frames = Vec::new();
        backtrace::trace(|cx| {
            physical_frames.push(cx.ip());

            // We only want to capture this closure's frame, assert_cpp_frames,
            // space::templated_trampoline, and cpp_trampoline. Those are
            // logical frames, which might be inlined into fewer physical
            // frames, so we may end up with extra logical frames after
            // resolving these.
            physical_frames.len() < 5
        });

        let names: Vec<_> = physical_frames
            .into_iter()
            .flat_map(|ip| {
                let mut logical_frame_names = vec![];

                backtrace::resolve(ip, |sym| {
                    let sym_name = sym.name().expect("Should have a symbol name");
                    let demangled = sym_name.to_string();
                    logical_frame_names.push(demangled);
                });

                assert!(
                    !logical_frame_names.is_empty(),
                    "Should have resolved at least one symbol for the physical frame"
                );

                logical_frame_names
            })
            // Skip the backtrace::trace closure and assert_cpp_frames, and then
            // take the two C++ frame names.
            .skip_while(|name| !name.contains("trampoline"))
            .take(2)
            .collect();

        println!("actual names = {names:#?}");

        let expected =
            ["void space::templated_trampoline<void (*)()>(void (*)())", "cpp_trampoline"];
        println!("expected names = {expected:#?}");

        assert_eq!(names.len(), expected.len());
        for (actual, expected) in names.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }

        RAN_ASSERTS.store(true, Ordering::SeqCst);
    }

    assert!(!RAN_ASSERTS.load(Ordering::SeqCst));
    unsafe {
        cpp_trampoline(assert_cpp_frames);
    }
    assert!(RAN_ASSERTS.load(Ordering::SeqCst));
}
