#![allow(clippy::borrow_deref_ref)]
#![warn(clippy::cloned_ref_to_slice_refs)]

#[derive(Clone)]
struct Data;

fn main() {
    {
        let data = Data;
        let data_ref = &data;
        let _ = &[data_ref.clone()]; //~ cloned_ref_to_slice_refs
    }

    {
        let _ = &[Data.clone()]; //~ cloned_ref_to_slice_refs
    }

    {
        #[derive(Clone)]
        struct Point(i32, i32);

        let _ = &[Point(0, 0).clone()]; //~ cloned_ref_to_slice_refs
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

fn issue16320(items: &[String]) {
    use std::ffi::OsString;
    use std::ops::Deref;
    use std::path::PathBuf;

    let _a = String::new();
    let _b = &[_a.to_owned()];
    //~^ cloned_ref_to_slice_refs
    let _c = &[_a.to_string()];
    //~^ cloned_ref_to_slice_refs

    let _a = OsString::new();
    let _b = &[_a.to_os_string()];
    //~^ cloned_ref_to_slice_refs

    let _a = PathBuf::new();
    let _b = &[_a.to_path_buf()];
    //~^ cloned_ref_to_slice_refs

    let _a = &PathBuf::new();
    let _b = &[_a.to_path_buf()];
    //~^ cloned_ref_to_slice_refs

    #[derive(Clone)]
    struct A(i32);

    impl std::fmt::Display for A {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    let a = A(42);
    _ = &[a.to_string()];

    struct Wrapper<T>(T);
    impl<T> Deref for Wrapper<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    let w = Wrapper(String::from("hello"));
    let w = Wrapper(w);
    let _b = &[w.to_string()];
    //~^ cloned_ref_to_slice_refs

    let w = Wrapper(&PathBuf::new());
    let w = Wrapper(w);
    let _b = &[w.to_path_buf()];
    //~^ cloned_ref_to_slice_refs
}

fn wrongly_unmangled_macros(items: &[String]) {
    use std::path::PathBuf;

    struct Wrapper {
        inner: PathBuf,
    }

    let _a = Wrapper { inner: PathBuf::new() };

    macro_rules! accessor {
        ($e:expr) => {
            $e.inner
        };
    }

    let _d = &[accessor!(_a).to_path_buf()];
    //~^ cloned_ref_to_slice_refs
}
