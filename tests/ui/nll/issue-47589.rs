// run-pass

pub struct DescriptorSet<'a> {
    pub slots: Vec<AttachInfo<'a, Resources>>
}

pub trait ResourcesTrait<'r>: Sized {
    type DescriptorSet: 'r;
}

pub struct Resources;

impl<'a> ResourcesTrait<'a> for Resources {
    type DescriptorSet = DescriptorSet<'a>;
}

pub enum AttachInfo<'a, R: ResourcesTrait<'a>> {
    NextDescriptorSet(Box<R::DescriptorSet>)
}

fn main() {
    let _x = DescriptorSet {slots: Vec::new()};
}
