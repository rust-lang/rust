// See coherence_inherent_cc.rs

pub trait TheTrait {
    fn the_fn(&self);
}

pub struct TheStruct;

impl TheTrait for TheStruct {
    fn the_fn(&self) {}
}
