// ICE unexpected inference var
// issue: rust-lang/rust#116599
//@ check-pass

pub trait EvaluateConstMethods {
    type Trait: TraitWithConstMethods;

    /// **This block breaks**
    const DATA_3: Data3 = {
        <<<Self::Trait as TraitWithConstMethods>::Method2 as ConstFn<_, _>>::Body<
            <<Self::Trait as TraitWithConstMethods>::Method1 as ConstFn<_, _>>::Body<ContainsData1>,
        > as Contains<_>>::ITEM
    };
}

pub trait TraitWithConstMethods {
    /// "const trait method" of signature `fn(Data1) -> Data2`
    type Method1: ConstFn<Data1, Data2>;

    /// "const trait method" of signature `fn(Data2) -> Data3`
    type Method2: ConstFn<Data2, Data3>;
}

/// A trait which tries to implement const methods in traits
pub trait ConstFn<Arg, Ret> {
    type Body<T: Contains<Arg>>: Contains<Ret>;
}

/// A ZST which represents / "contains" a const value which can be pass to a [`ConstFn`]
pub trait Contains<T> {
    const ITEM: T;
}

pub struct ContainsData1;
impl Contains<Data1> for ContainsData1 {
    const ITEM: Data1 = Data1 {};
}

// Arbitrary data
pub struct Data1 {}
pub struct Data2 {}
pub struct Data3 {}

pub fn main() {}
