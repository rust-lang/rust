//@ edition:2015
trait MyIterator : Iterator {}

fn main() {
    let _ = MyIterator::next;
}
//~^^ ERROR the value of the associated type `Item` in `Iterator` must be specified [E0191]
//~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
