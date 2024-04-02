// Check that nested statics in thread locals are
// duplicated per thread.

#![feature(const_refs_to_cell)]
#![feature(thread_local)]

//@run-pass

#[thread_local]
static mut FOO: &mut u32 = &mut 42;

fn main() {
    unsafe {
        *FOO = 1;

        let _ = std::thread::spawn(|| {
            assert_eq!(*FOO, 42);
            *FOO = 99;
        })
        .join();

        assert_eq!(*FOO, 1);
    }
}
