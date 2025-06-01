#![warn(clippy::cloned_ref_to_slice_refs)]

#[derive(Clone)]
struct Data;

fn main() {
    {
        let data = Data;
        let data_ref = &data;
        let _ = &[data_ref.clone()]; //~ ERROR: this call to `clone` can be replaced with `std::slice::from_ref`
    }

    {
        let _ = &[Data.clone()]; //~ ERROR: this call to `clone` can be replaced with `std::slice::from_ref`
    }

    {
        #[derive(Clone)]
        struct Point(i32, i32);

        let _ = &[Point(0, 0).clone()]; //~ ERROR: this call to `clone` can be replaced with `std::slice::from_ref`
    }

    // the string was cloned with the intention to not mutate
    {
        struct BetterString(String);

        let mut message = String::from("good");
        let sender = BetterString(message.clone());

        message.push_str("bye!");

        println!("{} {}", message, sender.0)
    }

    // the string was cloned with the intention to not mutate
    {
        let mut x = String::from("Hello");
        let r = &[x.clone()];
        x.push('!');
        println!("r = `{}', x = `{x}'", r[0]);
    }

    // mutable borrows may have the intention to clone
    {
        let data = Data;
        let data_ref = &data;
        let _ = &mut [data_ref.clone()];
    }

    // `T::clone` is used to denote a clone with side effects
    {
        use std::sync::Arc;
        let data = Arc::new(Data);
        let _ = &[Arc::clone(&data)];
    }

    // slices with multiple members can only be made from a singular reference
    {
        let data_1 = Data;
        let data_2 = Data;
        let _ = &[data_1.clone(), data_2.clone()];
    }
}
