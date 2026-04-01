#![allow(unused_assignments)]
//@ failure-status: 1

struct Firework {
    strength: i32,
}

impl Drop for Firework {
    fn drop(&mut self) {
        println!("BOOM times {}!!!", self.strength);
    }
}

fn main() -> Result<(), u8> {
    let _firecracker = Firework { strength: 1 };

    let _tnt = Firework { strength: 100 };

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
