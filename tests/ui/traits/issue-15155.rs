//@ run-pass
trait TraitWithSend: Send {}
trait IndirectTraitWithSend: TraitWithSend {}

// Check struct instantiation (Box<TraitWithSend> will only have Send if TraitWithSend has Send)
#[allow(dead_code)]
struct Blah { x: Box<dyn TraitWithSend> }
impl TraitWithSend for Blah {}

// Struct instantiation 2-levels deep
#[allow(dead_code)]
struct IndirectBlah { x: Box<dyn IndirectTraitWithSend> }
impl TraitWithSend for IndirectBlah {}
impl IndirectTraitWithSend for IndirectBlah {}

fn test_trait<T: Send + ?Sized>() { println!("got here!") }

fn main() {
    test_trait::<dyn TraitWithSend>();
    test_trait::<dyn IndirectTraitWithSend>();
}
