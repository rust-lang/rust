// Check that ordering of errors is correctly reported even with consts preceding types.

#![feature(const_generics)]
#![allow(incomplete_features)]

struct Example<'a, const N: usize, T=f32> {
  s: &'a T,
}

type Consts = Example<3, 3, 3>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of const arguments
type Types = Example<f32, f32, f32>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of const arguments
//~| ERROR wrong number of type arguments
type Lifetimes = Example<'static, 'static, 'static>;
//~^ ERROR wrong number of const arguments
//~| ERROR misplaced type arguments
//~| wrong number of lifetime

type LtConst1 = Example<'static, 3, 3>;
//~^ ERROR wrong number of const arguments
type LtConst2 = Example<3, 'static, 3>;
//~^ ERROR wrong number of const arguments
type LtConst3 = Example<3, 3, 'static>;
//~^ ERROR misplaced type arguments

type LtTy1 = Example<'static, f32, f32>;
//~^ ERROR wrong number of const arguments
//~| ERROR wrong number of type arguments
type LtTy2 = Example<f32, 'static, f32>;
//~^ ERROR wrong number of const arguments
//~| ERROR wrong number of type arguments
type LtTy3 = Example<f32, f32, 'static>;
//~^ ERROR wrong number of const arguments
//~| ERROR wrong number of type arguments

type ConstTy1 = Example<3, f32, f32>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of type arguments
type ConstTy2 = Example<f32, 3, f32>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of type arguments
type ConstTy3 = Example<f32, f32, 3>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of type arguments

type ConstLt1 = Example<3, 'static, 'static>;
//~^ ERROR wrong number of lifetime
type ConstLt2 = Example<'static, 3, 'static>;
//~^ ERROR wrong number of lifetime
type ConstLt3 = Example<'static, 'static, 3>;
//~^ ERROR wrong number of lifetime

type TyLt1 = Example<f32, 'static, 'static>;
//~^ ERROR wrong number of lifetime
//~| ERROR wrong number of const
//~| ERROR misplaced type arguments
type TyLt2 = Example<'static, f32, 'static>;
//~^ ERROR wrong number of lifetime
//~| ERROR wrong number of const
//~| ERROR misplaced type arguments
type TyLt3 = Example<'static, 'static, f32>;
//~^ ERROR wrong number of const
//~| ERROR wrong number of lifetime

type TyConst1 = Example<f32, 3, 3>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of const
//~| ERROR misplaced type arguments
type TyConst2 = Example<3, f32, 3>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of const
type TyConst3 = Example<3, 3, f32>;
//~^ ERROR missing lifetime specifier
//~| ERROR wrong number of const
//~| ERROR misplaced type arguments

type Intermixed1 = Example<'static, 3, f32>; // ok


type Intermixed2 = Example<f32, 'static, 3>;
//~^ ERROR type provided when a constant was expected
type Intermixed3 = Example<3, f32, 'static>;
//~^ ERROR constant provided when a lifetime

fn main() {}
