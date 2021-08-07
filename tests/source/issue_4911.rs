#![feature(generic_associated_types)]
#![feature(min_type_alias_impl_trait)]

impl SomeTrait for SomeType {
    type SomeGAT<'a> where Self: 'a = impl SomeOtherTrait;
}