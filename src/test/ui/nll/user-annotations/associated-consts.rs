struct MyTy<T>(T);
impl<T> MyTy<T> {
    const INHERENT: bool = true;
}

trait Trait {
    const TRAIT: bool;
}
impl<T> Trait for MyTy<T> {
    const TRAIT: bool = true;
}

fn test<'a, 'b>() {
    MyTy::<&'static &'a ()>::INHERENT; //~ ERROR lifetime
    MyTy::<&'static &'b ()>::TRAIT; //~ ERROR lifetime
}

fn test_normalization<'a, 'b>() {
    trait Project {
        type Assoc;
    }
    impl<T: 'static> Project for T {
        type Assoc = &'static &'static ();
    }
    MyTy::<<&'a () as Project>::Assoc>::INHERENT; //~ ERROR lifetime
    MyTy::<<&'b () as Project>::Assoc>::TRAIT; //~ ERROR lifetime
}

fn main() {}
