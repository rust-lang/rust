// run-pass
#![allow(dead_code)]

// Don't panic on blocks without results
// There are several tests in this run-pass that raised
// when this bug was opened. The cases where the compiler
// panics before the fix have a comment.

struct S {x:()}

fn test(slot: &mut Option<Box<dyn FnMut() -> Box<dyn FnMut()>>>) -> () {
  let a = slot.take();
  let _a: () = match a {
    // `{let .. a(); }` would break
    Some(mut a) => { let _a = a(); },
    None => (),
  };
}

fn not(b: bool) -> bool {
    if b {
        !b
    } else {
        // `panic!(...)` would break
        panic!("Break the compiler");
    }
}

pub fn main() {
    // {} would break
    let _r: () = {};
    let mut slot = None;
    // `{ test(...); }` would break
    let _s : S  = S{ x: { test(&mut slot); } };

    let _b = not(true);
}
