//@ run-pass
//@ edition: 2024

fn main() {
    let mut x = 0;

    match 3_i32 {
        i32 { .. } => x += 1,
    }

    match "hello world" {
        str { .. } => x += 1,
    }

    type Unit = ();

    match () {
        Unit { .. } => x += 1,
    }

    match &3_i32 {
        &i32 { .. } => x += 1,
    }

    match &3_i32 {
        i32 { .. } => x += 1,
    }

    type Foo = dyn Send;
    let a: &Foo = &3_i32;
    match a {
        &Foo { .. } => x += 1,
    }

    match Some(false) {
        Option { .. } => x += 1,
    }

    fn foo<T>(input: T, x: &mut usize) {
        match input {
            T { .. } => *x += 1,
        }
    }

    foo(&&(18, 32, false, "abcdefg"), &mut x);

    trait Trait {
        type Assoc;

        fn method(&self, assoc: Self::Assoc, x: &mut usize) {
            match self {
                Self { .. } => *x += 1,
            }

            match assoc {
                Self::Assoc { .. } => *x += 1,
            }
        }
    }

    impl Trait for (u64,) {
        type Assoc = [f32; 17];
    }

    (73_u64,).method([4.2; 17], &mut x);

    assert_eq!(x, 10);
}
