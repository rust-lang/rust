import libc, unsafe;

enum malloc_pool = ();

impl methods for malloc_pool {
    fn alloc(sz: uint, align: uint) -> *() {
        unsafe {
            unsafe::reinterpret_cast(libc::malloc(sz as libc::size_t))
        }
    }
}

fn main() {
    let p = &malloc_pool(());
    let x = new(*p) 4u;
    io::print(#fmt["%u", *x]);
    assert *x == 4u;
    unsafe {
        libc::free(unsafe::reinterpret_cast(x));
    }
}
