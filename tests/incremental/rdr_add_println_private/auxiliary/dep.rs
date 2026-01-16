// Auxiliary crate for testing that adding println! to a private function
// does not cause dependent crates to rebuild.
//
// This is important for RDR because println! expands to code with spans,
// and those spans should not leak into the public API metadata.

#![crate_name = "dep"]
#![crate_type = "rlib"]

// Public API - unchanged across all revisions
pub fn public_fn(x: u32) -> u32 {
    private_helper(x)
}

pub struct PublicStruct {
    pub value: u32,
}

impl PublicStruct {
    pub fn compute(&self) -> u32 {
        private_compute(self.value)
    }
}

// Private implementation

// rpass1: No println
#[cfg(rpass1)]
fn private_helper(x: u32) -> u32 {
    x + 1
}

// rpass2: Add println! to private function
#[cfg(rpass2)]
fn private_helper(x: u32) -> u32 {
    println!("private_helper called with {}", x);
    x + 1
}

// rpass3: Add more println! statements
#[cfg(rpass3)]
fn private_helper(x: u32) -> u32 {
    println!("private_helper called with {}", x);
    println!("computing result...");
    let result = x + 1;
    println!("result is {}", result);
    result
}

// rpass1: No println
#[cfg(rpass1)]
fn private_compute(x: u32) -> u32 {
    x * 2
}

// rpass2: Add eprintln! to different private function
#[cfg(rpass2)]
fn private_compute(x: u32) -> u32 {
    eprintln!("private_compute: {}", x);
    x * 2
}

// rpass3: Add debug formatting
#[cfg(rpass3)]
fn private_compute(x: u32) -> u32 {
    eprintln!("private_compute input: {:?}", x);
    let result = x * 2;
    eprintln!("private_compute output: {:?}", result);
    result
}
