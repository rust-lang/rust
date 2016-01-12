extern crate thread_local_extern_static;

extern {
    #[thread_local]
    static FOO: u32;
}

fn main() {
    assert_eq!(FOO, 3);
}
