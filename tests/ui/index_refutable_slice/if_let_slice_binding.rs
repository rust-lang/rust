#![deny(clippy::index_refutable_slice)]

enum SomeEnum<T> {
    One(T),
    Two(T),
    Three(T),
    Four(T),
}

fn lintable_examples() {
    // Try with reference
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice) = slice {
        println!("{}", slice[0]);
    }

    // Try with copy
    let slice: Option<[u32; 3]> = Some([1, 2, 3]);
    if let Some(slice) = slice {
        println!("{}", slice[0]);
    }

    // Try with long slice and small indices
    let slice: Option<[u32; 9]> = Some([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    if let Some(slice) = slice {
        println!("{}", slice[2]);
        println!("{}", slice[0]);
    }

    // Multiple bindings
    let slice_wrapped: SomeEnum<[u32; 3]> = SomeEnum::One([5, 6, 7]);
    if let SomeEnum::One(slice) | SomeEnum::Three(slice) = slice_wrapped {
        println!("{}", slice[0]);
    }

    // Two lintable slices in one if let
    let a_wrapped: SomeEnum<[u32; 3]> = SomeEnum::One([9, 5, 1]);
    let b_wrapped: Option<[u32; 2]> = Some([4, 6]);
    if let (SomeEnum::Three(a), Some(b)) = (a_wrapped, b_wrapped) {
        println!("{} -> {}", a[2], b[1]);
    }

    // This requires the slice values to be borrowed as the slice values can only be
    // borrowed and `String` doesn't implement copy
    let slice: Option<[String; 2]> = Some([String::from("1"), String::from("2")]);
    if let Some(ref slice) = slice {
        println!("{:?}", slice[1]);
    }
    println!("{:?}", slice);

    // This should not suggest using the `ref` keyword as the scrutinee is already
    // a reference
    let slice: Option<[String; 2]> = Some([String::from("1"), String::from("2")]);
    if let Some(slice) = &slice {
        println!("{:?}", slice[0]);
    }
    println!("{:?}", slice);
}

fn slice_index_above_limit() {
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);

    if let Some(slice) = slice {
        // Would cause a panic, IDK
        println!("{}", slice[7]);
    }
}

fn slice_is_used() {
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice) = slice {
        println!("{:?}", slice.len());
    }

    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice) = slice {
        println!("{:?}", slice.to_vec());
    }

    let opt: Option<[String; 2]> = Some([String::from("Hello"), String::from("world")]);
    if let Some(slice) = opt {
        if !slice.is_empty() {
            println!("first: {}", slice[0]);
        }
    }
}

/// The slice is used by an external function and should therefore not be linted
fn check_slice_as_arg() {
    fn is_interesting<T>(slice: &[T; 2]) -> bool {
        !slice.is_empty()
    }

    let slice_wrapped: Option<[String; 2]> = Some([String::from("Hello"), String::from("world")]);
    if let Some(slice) = &slice_wrapped {
        if is_interesting(slice) {
            println!("This is interesting {}", slice[0]);
        }
    }
    println!("{:?}", slice_wrapped);
}

fn check_slice_in_struct() {
    #[derive(Debug)]
    struct Wrapper<'a> {
        inner: Option<&'a [String]>,
        is_awesome: bool,
    }

    impl<'a> Wrapper<'a> {
        fn is_super_awesome(&self) -> bool {
            self.is_awesome
        }
    }

    let inner = &[String::from("New"), String::from("World")];
    let wrap = Wrapper {
        inner: Some(inner),
        is_awesome: true,
    };

    // Test 1: Field access
    if let Some(slice) = wrap.inner {
        if wrap.is_awesome {
            println!("This is awesome! {}", slice[0]);
        }
    }

    // Test 2: function access
    if let Some(slice) = wrap.inner {
        if wrap.is_super_awesome() {
            println!("This is super awesome! {}", slice[0]);
        }
    }
    println!("Complete wrap: {:?}", wrap);
}

/// This would be a nice additional feature to have in the future, but adding it
/// now would make the PR too large. This is therefore only a test that we don't
/// lint cases we can't make a reasonable suggestion for
fn mutable_slice_index() {
    // Mut access
    let mut slice: Option<[String; 1]> = Some([String::from("Penguin")]);
    if let Some(ref mut slice) = slice {
        slice[0] = String::from("Mr. Penguin");
    }
    println!("Use after modification: {:?}", slice);

    // Mut access on reference
    let mut slice: Option<[String; 1]> = Some([String::from("Cat")]);
    if let Some(slice) = &mut slice {
        slice[0] = String::from("Lord Meow Meow");
    }
    println!("Use after modification: {:?}", slice);
}

/// The lint will ignore bindings with sub patterns as it would be hard
/// to build correct suggestions for these instances :)
fn binding_with_sub_pattern() {
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice @ [_, _, _]) = slice {
        println!("{:?}", slice[2]);
    }
}

fn main() {}
