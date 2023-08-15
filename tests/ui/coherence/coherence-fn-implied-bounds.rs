trait Trait {}

impl Trait for for<'a, 'b> fn(&'a &'b u32, &'b &'a u32) -> &'b u32 {}

impl Trait for for<'c> fn(&'c &'c u32, &'c &'c u32) -> &'c u32 {
    //~^ ERROR conflicting implementations
}

fn main() {}
