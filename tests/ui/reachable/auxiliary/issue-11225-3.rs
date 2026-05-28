trait PrivateTrait {
    fn private_trait_method(&self);
    fn private_trait_method_ufcs(&self);
}

struct PrivateStruct;

impl PrivateStruct {
    fn private_inherent_method(&self) { }
    fn private_inherent_method_ufcs(&self) { }
}

impl PrivateTrait for PrivateStruct {
    fn private_trait_method(&self) { }
    fn private_trait_method_ufcs(&self) { }
}

#[inline]
pub fn public_inlinable_function() {
    PrivateStruct.private_trait_method();
    PrivateStruct.private_inherent_method();
}

#[inline]
pub fn public_inlinable_function_ufcs() {
    PrivateStruct::private_trait_method(&PrivateStruct);
    PrivateStruct::private_inherent_method(&PrivateStruct);
}
