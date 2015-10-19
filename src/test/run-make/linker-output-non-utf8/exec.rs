#[link(name="library")]
extern "C" {
    fn foo();
}

fn main() { unsafe { foo(); } }
