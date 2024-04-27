//@ known-bug: #124342
trait Trait2 : Trait {
   reuse <() as Trait>::async {
        (async || {}).await;
    };
}
