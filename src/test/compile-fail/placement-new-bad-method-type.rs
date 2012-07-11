import libc, unsafe;

enum malloc_pool = ();

trait alloc {
    fn alloc(sz: int, align: int) -> *();
}

impl methods of alloc for malloc_pool {
    fn alloc(sz: int, align: int) -> *() {
        fail;
    }
}

fn main() {
    let p = &malloc_pool(());
    let x = new(*p) 4u;
    //~^ ERROR mismatched types: expected `fn(*()) -> *()`
}
