
import rustrt::size_of;

export rustrt;
export size_of;

native "rust" mod rustrt {

        // Explicitly re-export native stuff we want to be made
        // available outside this crate. Otherwise it's
        // visible-in-crate, but not re-exported.
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
