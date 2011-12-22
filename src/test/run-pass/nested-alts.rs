
fn baz() -> ! { fail; }

fn foo() {
    alt some::<int>(5) {
      some::<int>(x) {
        let bar;
        alt none::<int> { none::<int>. { bar = 5; } _ { baz(); } }
        log_full(core::debug, bar);
      }
      none::<int>. { #debug("hello"); }
    }
}

fn main() { foo(); }
