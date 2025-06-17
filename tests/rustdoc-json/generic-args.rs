pub struct MyStruct(u32);

pub trait MyTrait {
    type MyType;
    fn my_fn(&self);
}

impl MyTrait for MyStruct {
    type MyType = u32;
    fn my_fn(&self) {}
}

//@ is "$.index[?(@.name=='my_fn1')].inner.function.sig.inputs[0][1].qualified_path.args" {\"angle_bracketed\":{\"args\":[],\"constraints\":[]}}
//@ is "$.index[?(@.name=='my_fn1')].inner.function.sig.inputs[0][1].qualified_path.self_type.resolved_path.args" {\"angle_bracketed\":{\"args\":[],\"constraints\":[]}}
pub fn my_fn1(_: <MyStruct as MyTrait>::MyType) {}

//@ is "$.index[?(@.name=='my_fn2')].inner.function.sig.inputs[0][1].dyn_trait.traits[0].trait.args.angle_bracketed.constraints[0].args" {\"angle_bracketed\":{\"args\":[],\"constraints\":[]}}
pub fn my_fn2(_: IntoIterator<Item = MyStruct, IntoIter = impl Clone>) {}

fn main() {}
