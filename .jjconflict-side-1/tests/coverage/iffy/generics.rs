#![allow(unused_assignments)]
//@ failure-status: 1

struct Firework<T: Copy + std::fmt::Display> {
    strength: T,
}

impl<T: Copy + std::fmt::Display> Firework<T> {
    #[inline(always)]
    fn set_strength(&mut self, new_strength: T) {
        self.strength = new_strength;
    }
}

impl<T: Copy + std::fmt::Display> Drop for Firework<T> {
    #[inline(always)]
    fn drop(&mut self) {
        println!("BOOM times {}!!!", self.strength);
    }
}

fn main() -> Result<(), u8> {
    let mut firecracker = Firework { strength: 1 };
    firecracker.set_strength(2);

    let mut tnt = Firework { strength: 100.1 };
    tnt.set_strength(200.1);
    tnt.set_strength(300.3);

    if true {
        println!("Exiting with error...");
        return Err(1);
    }

    let _ = Firework { strength: 1000 };

    Ok(())
}

// Expected program output:
//   Exiting with error...
//   BOOM times 100!!!
//   BOOM times 1!!!
//   Error: 1
