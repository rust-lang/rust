// reported as #124164
static S_COUNT: = std::sync::atomic::AtomicUsize::new(0);
//~^ ERROR: missing type for `static` item

fn main() {}
