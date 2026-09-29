fn main() {
    let target: Target = create_target();
    target.unique_name(0); // correct arguments work
    target.unique_name(10.0); // (used to crash here)
    //~^ ERROR mismatched types
}

// must be generic
fn create_target<T>() -> T {
    unimplemented!()
}

// unimplemented trait, but contains function with the same name
pub trait RandomTrait {
    fn unique_name(&mut self); // but less arguments
}

struct Target;

impl Target {
    // correct function with arguments
    pub fn unique_name(&self, data: i32) {
        unimplemented!()
    }
}
