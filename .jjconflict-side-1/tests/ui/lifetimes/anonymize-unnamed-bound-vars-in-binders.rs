//@ build-pass
// issue: #115807

trait Chip: for<'a> TraitWithLifetime<'a> + SomeMarker {
    fn compute(&self);
}

trait SomeMarker {}

trait TraitWithLifetime<'a>: SomeMarker {}

trait Machine {
    fn run();
}

struct BasicMachine;

impl Machine for BasicMachine {
    fn run() {
        let chips: [&dyn Chip; 0] = [];
        let _ = chips.map(|chip| chip.compute());
    }
}

fn main() {
    BasicMachine::run();
}
