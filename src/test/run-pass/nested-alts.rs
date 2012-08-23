
fn baz() -> ! { fail; }

fn foo() {
    match some::<int>(5) {
      some::<int>(x) => {
        let mut bar;
        match none::<int> { none::<int> => { bar = 5; } _ => { baz(); } }
        log(debug, bar);
      }
      none::<int> => { debug!("hello"); }
    }
}

fn main() { foo(); }
