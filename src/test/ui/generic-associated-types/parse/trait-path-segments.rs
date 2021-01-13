#![feature(generic_associated_types)]
//~^ WARNING: the feature `generic_associated_types` is incomplete

const _: () = {
    trait X {
        type Y<'a>;
    }

    fn f1<'a>(arg : Box<dyn X<X::Y = u32>>) {}
        //~^ ERROR: paths with multiple segments cannot be used in associated type constraints
  };

const _: () = {
    trait X {
        type Y<'a>;
    }

    trait Z {}

    impl<T : X<<Self as X>::Y<'a> = &'a u32>> Z for T {}
        //~^ ERROR: qualified paths cannot be used in associated type constraints
};

const _: () = {
    trait X {
      type Y<'a>;
    }

    trait Z {}

    impl<T : X<X::Y<'a> = &'a u32>> Z for T {}
        //~^ ERROR: paths with multiple segments cannot be used in associated type constraints
};

fn main() {}
