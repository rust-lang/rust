#![feature(explicit_tail_calls, exclusive_wrapper, fn_traits, unboxed_closures)]
#![expect(incomplete_features)]

fn f() {}

fn g() {
    become std::sync::Exclusive::new(f)() //~ error: tail calls can only be performed with function definitions or pointers
}

fn h() {
    become (&mut &std::sync::Exclusive::new(f))() //~ error: tail calls can only be performed with function definitions or pointers
}

fn i() {
    struct J;

    impl FnOnce<()> for J {
        type Output = ();
        extern "rust-call" fn call_once(self, (): ()) -> Self::Output {}
    }

    become J(); //~ error: tail calls can only be performed with function definitions or pointers
}

fn main() {
    g();
    h();
    i();
}
