use std::fmt::Debug;

use super::Bridge;

pub trait SmirError {
    fn new(msg: String) -> Self;
    fn from_internal<T: Debug>(err: T) -> Self;
}

macro_rules! make_def_trait {
    ($name:ident) => {
        pub trait $name<B: Bridge> {
            fn new(did: B::DefId) -> Self;
        }
    };
}

make_def_trait!(CrateItem);
make_def_trait!(AdtDef);
make_def_trait!(ForeignModuleDef);
make_def_trait!(ForeignDef);
make_def_trait!(FnDef);
make_def_trait!(ClosureDef);
make_def_trait!(CoroutineDef);
make_def_trait!(CoroutineClosureDef);
make_def_trait!(AliasDef);
make_def_trait!(ParamDef);
make_def_trait!(BrNamedDef);
make_def_trait!(TraitDef);
make_def_trait!(GenericDef);
make_def_trait!(ConstDef);
make_def_trait!(ImplDef);
make_def_trait!(RegionDef);
make_def_trait!(CoroutineWitnessDef);
make_def_trait!(AssocDef);
make_def_trait!(OpaqueDef);
make_def_trait!(StaticDef);

pub trait Prov<B: Bridge> {
    fn new(aid: B::AllocId) -> Self;
}
