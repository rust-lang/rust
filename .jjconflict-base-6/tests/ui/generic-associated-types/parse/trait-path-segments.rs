const _: () = {
    trait X {
        type Y<'a>;
    }

    fn f1<'a>(arg : Box<dyn X<X::Y = u32>>) {}
        //~^ ERROR: expected one of
  };

const _: () = {
    trait X {
        type Y<'a>;
    }

    trait Z {}

    impl<T : X<<Self as X>::Y<'a> = &'a u32>> Z for T {}
        //~^ ERROR: expected one of
};

const _: () = {
    trait X {
      type Y<'a>;
    }

    trait Z {}

    impl<T : X<X::Y<'a> = &'a u32>> Z for T {}
        //~^ ERROR: expected one of
};

fn main() {}
