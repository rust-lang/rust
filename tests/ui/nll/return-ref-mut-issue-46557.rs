// Regression test for issue #46557

fn gimme_static_mut() -> &'static mut u32 {
    let ref mut x = 1234543;
    x //~ ERROR cannot return value referencing temporary value [E0515]
}

fn main() {}
