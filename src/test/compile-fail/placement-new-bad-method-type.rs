import libc, unsafe;

enum malloc_pool = ();

impl methods for malloc_pool {
    fn alloc(sz: int, align: int) -> *() {
        fail;
    }
}

fn main() {
    let p = &malloc_pool(());
    let x = new(*p) 4u;
    //!^ ERROR mismatched types: expected `fn(uint, uint) -> *()`
}
