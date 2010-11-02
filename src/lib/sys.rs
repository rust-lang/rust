export rustrt;

native "rust" mod rustrt {

    // Explicitly re-export native stuff we want to be made
    // available outside this crate. Otherwise it's
    // visible-in-crate, but not re-exported.

    export last_os_error;
    export size_of;
    export align_of;
    export refcount;
    export do_gc;

    fn last_os_error() -> str;
    fn size_of[T]() -> uint;
    fn align_of[T]() -> uint;
    fn refcount[T](@T t) -> uint;
    fn do_gc();
    fn unsupervise();
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
