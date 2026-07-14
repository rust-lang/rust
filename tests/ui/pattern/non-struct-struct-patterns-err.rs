//@ edition: 2024

fn main() {
    let mut x = 0;

    match 3_i32 {
        u64 { .. } => x += 1,
        //~^ ERROR mismatched types
    }

    match "hello world" {
        char { .. } => x += 1,
        //~^ ERROR mismatched types
    }

    type Unit = ();

    match (false,) {
        Unit { .. } => x += 1,
        //~^ ERROR mismatched types
    }

    match &3_i32 {
        &&&&i32 { .. } => x += 1,
        //~^ ERROR mismatched types
    }

    match Some(false) {
        Result { .. } => x += 1,
        //~^ ERROR mismatched types
    }

    fn foo<T, U>(input: T, input_2: U, x: &mut usize) {
        match input {
            U { .. } => *x += 1,
            //~^ ERROR mismatched types
        }

        match input_2 {
            T { .. } => *x += 1,
            //~^ ERROR mismatched types
        }
    }

    foo(&&(18, 32, false, "abcdefg"), true, &mut x);

    trait Trait {
        type Assoc;

        fn method(&self, assoc: Self::Assoc, x: &mut usize) {
            match self {
                Self::Assoc { .. } => *x += 1,
                //~^ ERROR mismatched types
            }

            match assoc {
                Self { .. } => *x += 1,
                //~^ ERROR mismatched types
            }
        }
    }

    impl Trait for (u64,) {
        type Assoc = [f32; 17];
    }

    (73_u64,).method([4.2; 17], &mut x);

    assert_eq!(x, 10);
}
