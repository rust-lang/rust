// check-pass
// compile-flags: -C opt-level=3

#![crate_type = "lib"]

pub trait Backend {
    type DescriptorSetLayout;
}

pub struct Back;

impl Backend for Back {
    type DescriptorSetLayout = u32;
}

pub struct HalSetLayouts {
    vertex_layout: <Back as Backend>::DescriptorSetLayout,
}

impl HalSetLayouts {
    pub fn iter<DSL>(self) -> DSL
    where
        Back: Backend<DescriptorSetLayout = DSL>,
    {
        self.vertex_layout
    }
}
