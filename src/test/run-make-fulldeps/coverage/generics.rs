#![allow(unused_assignments)]
// expect-exit-status-1

struct Firework<T> where T: Copy + std::fmt::Display {
    strength: T,
}

impl<T> Firework<T> where T: Copy + std::fmt::Display {
    #[inline(always)]
    fn set_strength(&mut self, new_strength: T) {
        self.strength = new_strength;
    }
}

impl<T> Drop for Firework<T> where T: Copy + std::fmt::Display {
    #[inline(always)]
    fn drop(&mut self) {
        println!("BOOM times {}!!!", self.strength);
    }
}

fn main() -> Result<(),u8> {
    let mut firecracker = Firework { strength: 1 };
    firecracker.set_strength(2);

    let mut tnt = Firework { strength: 100.1 };
    tnt.set_strength(200.1);
    tnt.set_strength(300.3);

    if true {
        println!("Exiting with error...");
        return Err(1);
    } // The remaining lines below have no coverage because `if true` (with the constant literal
      // `true`) is guaranteed to execute the `then` block, which is also guaranteed to `return`.
      // Thankfully, in the normal case, conditions are not guaranteed ahead of time, and as shown
      // in other tests, the lines below would have coverage (which would show they had `0`
      // executions, assuming the condition still evaluated to `true`).

    let _ = Firework { strength: 1000 };

    Ok(())
}

// Expected program output:
//   Exiting with error...
//   BOOM times 100!!!
//   BOOM times 1!!!
//   Error: 1
