
//export rustrt;
//export size_of;

native "rust" mod rustrt {
    // Explicitly re-export native stuff we want to be made
    // available outside this crate. Otherwise it's
    // visible-in-crate, but not re-exported.
    fn last_os_error() -> str;
    fn size_of<T>() -> uint;
    fn align_of<T>() -> uint;
    fn refcount<T>(t: @T) -> uint;
    fn do_gc();
    fn unsupervise();
}

fn last_os_error() -> str {
    //unsafe {
        ret rustrt::last_os_error();
    //}
}

fn size_of<T>() -> uint {
    //unsafe {
        ret rustrt::size_of::<T>();
    //}
}

fn align_of<T>() -> uint {
    //unsafe {
        ret rustrt::align_of::<T>();
    //}
}

fn refcount<T>(t: @T) -> uint {
    //unsafe {
        ret rustrt::refcount::<T>(t);
    //}
}

fn do_gc() -> () {
    //unsafe {
        ret rustrt::do_gc();
    //}
}

fn unsupervise() -> () {
    //unsafe {
        ret rustrt::unsupervise();
    //}
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
