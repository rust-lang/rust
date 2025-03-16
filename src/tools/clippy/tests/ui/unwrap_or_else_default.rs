#![warn(clippy::unwrap_or_default)]
#![allow(dead_code)]
#![allow(clippy::unnecessary_wraps, clippy::unnecessary_literal_unwrap)]

/// Checks implementation of the `UNWRAP_OR_DEFAULT` lint.
fn unwrap_or_else_default() {
    struct Foo;

    impl Foo {
        fn new() -> Foo {
            Foo
        }

        // fake default, we should not trigger on this
        fn default() -> Foo {
            Foo
        }
    }

    struct HasDefaultAndDuplicate;

    impl HasDefaultAndDuplicate {
        fn default() -> Self {
            HasDefaultAndDuplicate
        }
    }

    impl Default for HasDefaultAndDuplicate {
        fn default() -> Self {
            HasDefaultAndDuplicate
        }
    }

    enum Enum {
        A(),
    }

    fn make<T, V>(_: V) -> T {
        unimplemented!();
    }

    let with_enum = Some(Enum::A());
    with_enum.unwrap_or_else(Enum::A);

    let with_new = Some(vec![1]);
    with_new.unwrap_or_else(Vec::new);
    //~^ unwrap_or_default

    let with_err: Result<_, ()> = Ok(vec![1]);
    with_err.unwrap_or_else(make);

    // should not be changed
    let with_fake_default = None::<Foo>;
    with_fake_default.unwrap_or_else(Foo::default);

    // should not be changed
    let with_fake_default2 = None::<HasDefaultAndDuplicate>;
    with_fake_default2.unwrap_or_else(<HasDefaultAndDuplicate>::default);

    let with_real_default = None::<HasDefaultAndDuplicate>;
    with_real_default.unwrap_or_else(<HasDefaultAndDuplicate as Default>::default);
    //~^ unwrap_or_default

    let with_default_trait = Some(1);
    with_default_trait.unwrap_or_else(Default::default);
    //~^ unwrap_or_default

    let with_default_type = Some(1);
    with_default_type.unwrap_or_else(u64::default);
    //~^ unwrap_or_default

    let with_default_type: Option<Vec<u64>> = None;
    with_default_type.unwrap_or_else(Vec::new);
    //~^ unwrap_or_default

    let empty_string = None::<String>;
    empty_string.unwrap_or_else(|| "".to_string());
    //~^ unwrap_or_default
}

fn type_certainty(option: Option<Vec<u64>>) {
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option: std::option::Option<std::vec::Vec<u64>> = None;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option: Option<Vec<u64>> = None;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option = std::option::Option::<std::vec::Vec<u64>>::None;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option = Option::<Vec<u64>>::None;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option = std::option::Option::None::<std::vec::Vec<u64>>;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option = Option::None::<Vec<u64>>;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    let option = None::<Vec<u64>>;
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default

    // should not be changed: type annotation with infer, unconcretized initializer
    let option: Option<Vec<_>> = None;
    option.unwrap_or_else(Vec::new).push(1);

    // should not be changed: no type annotation, unconcretized initializer
    let option = Option::None;
    option.unwrap_or_else(Vec::new).push(1);

    // should not be changed: no type annotation, unconcretized initializer
    let option = None;
    option.unwrap_or_else(Vec::new).push(1);

    type Alias = Option<Vec<u32>>;
    let option: Alias = Option::<Vec<u32>>::Some(Vec::new());
    option.unwrap_or_else(Vec::new).push(1);
    //~^ unwrap_or_default
}

fn method_call_with_deref() {
    use std::cell::RefCell;
    use std::collections::HashMap;

    let cell = RefCell::new(HashMap::<u64, HashMap<u64, String>>::new());

    let mut outer_map = cell.borrow_mut();

    #[allow(unused_assignments)]
    let mut option = None;
    option = Some(0);

    let inner_map = outer_map.get_mut(&option.unwrap()).unwrap();

    let _ = inner_map.entry(0).or_insert_with(Default::default);
    //~^ unwrap_or_default
}

fn missing_suggested_method() {
    #[derive(Copy, Clone)]
    struct S<T>(T);

    impl<T> S<T> {
        fn or_insert_with(&mut self, default: impl FnOnce() -> T) -> &mut T {
            &mut self.0
        }

        fn or_insert(&mut self, default: T) -> &mut T {
            &mut self.0
        }

        fn unwrap_or_else(self, default: impl FnOnce() -> T) -> T {
            self.0
        }

        fn unwrap_or(self, default: T) -> T {
            self.0
        }
    }

    // Don't lint when or_default/unwrap_or_default do not exist on the type
    let mut s = S(1);
    s.or_insert_with(Default::default);
    s.or_insert(Default::default());
    s.unwrap_or_else(Default::default);
    s.unwrap_or(Default::default());
}

fn main() {}
