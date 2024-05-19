//@ known-bug: #124347
trait Trait: ToReuse {
    reuse Trait::lolno { &self.0 };
}
