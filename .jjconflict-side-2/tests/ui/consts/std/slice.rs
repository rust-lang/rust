//@ build-pass (FIXME(62277): could be check-pass?)

struct Wrap<T>(T);
unsafe impl<T> Send for Wrap<T> {}
unsafe impl<T> Sync for Wrap<T> {}

static FOO: Wrap<*const u32> = Wrap([42, 44, 46].as_ptr());
static BAR: Wrap<*const u8> = Wrap("hello".as_ptr());

fn main() {}
