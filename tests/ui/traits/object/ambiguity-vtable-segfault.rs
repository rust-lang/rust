// In this example below, we have two overlapping candidates for `dyn Q: Q`.
// Specifically, the user written impl for `<dyn Q as Mirror>::Assoc` and the
// built-in impl for object types. Since they differ by their region responses,
// the goal is ambiguous. This affects codegen since impossible obligations
// for method dispatch will lead to a segfault, since we end up emitting dummy
// call vtable offsets due to <https://github.com/rust-lang/rust/pull/136311>.

// Test for <https://github.com/rust-lang/rust/issues/141119>.

//@ run-pass

trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}

trait Q: 'static {
    fn q(&self);
}

impl Q for i32 {
    fn q(&self) { println!("i32"); }
}

impl Q for <dyn Q as Mirror>::Assoc where Self: 'static {
    fn q(&self) { println!("dyn Q"); }
}

fn foo<T: Q + ?Sized>(t: &T) {
    t.q();
}

fn main() {
    foo(&1 as &dyn Q);
}
