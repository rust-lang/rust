#![crate_type="lib"]

pub trait Trait {
    // the issue is sensitive to interning order - so use names
    // unlikely to appear in libstd.
    type Issue25467FooT;
    type Issue25467BarT;
}

pub type Object = Option<Box<dyn Trait<Issue25467FooT=(),Issue25467BarT=()>>>;
