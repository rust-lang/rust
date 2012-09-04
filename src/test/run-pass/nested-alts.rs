
fn baz() -> ! { fail; }

fn foo() {
    match Some::<int>(5) {
      Some::<int>(x) => {
        let mut bar;
        match None::<int> { None::<int> => { bar = 5; } _ => { baz(); } }
        log(debug, bar);
      }
      None::<int> => { debug!("hello"); }
    }
}

fn main() { foo(); }
