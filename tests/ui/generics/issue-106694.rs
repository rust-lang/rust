trait Trait {}

fn foo(_: impl &Trait) {}
//~^ ERROR expected a trait, found type

fn bar<T: &Trait>(_: T) {}
//~^ ERROR expected a trait, found type

fn partially_correct_impl(_: impl &*const &Trait + Copy) {}
//~^ ERROR expected a trait, found type

fn foo_bad(_: impl &BadTrait) {}
//~^ ERROR expected a trait, found type
//~^^ ERROR cannot find trait `BadTrait` in this scope

fn bar_bad<T: &BadTrait>(_: T) {}
//~^ ERROR expected a trait, found type
//~^^ ERROR cannot find trait `BadTrait` in this scope

fn partially_correct_impl_bad(_: impl &*const &BadTrait + Copy) {}
//~^ ERROR expected a trait, found type
//~^^ ERROR cannot find trait `BadTrait` in this scope

fn main() {}
