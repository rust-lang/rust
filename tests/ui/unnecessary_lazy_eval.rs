// run-rustfix
#![warn(clippy::unnecessary_lazy_eval)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::bind_instead_of_map)]

struct Deep(Option<u32>);

#[derive(Copy, Clone)]
struct SomeStruct {
    some_field: u32,
}

impl SomeStruct {
    fn return_some_field(&self) -> u32 {
        self.some_field
    }
}

fn some_call<T: Default>() -> T {
    T::default()
}

fn main() {
    let astronomers_pi = 10;
    let ext_str = SomeStruct { some_field: 10 };

    // Should lint - Option
    let mut opt = Some(42);
    let ext_opt = Some(42);
    let _ = opt.unwrap_or_else(|| 2);
    let _ = opt.unwrap_or_else(|| astronomers_pi);
    let _ = opt.unwrap_or_else(|| ext_str.some_field);
    let _ = opt.and_then(|_| ext_opt);
    let _ = opt.or_else(|| ext_opt);
    let _ = opt.or_else(|| None);
    let _ = opt.get_or_insert_with(|| 2);
    let _ = opt.ok_or_else(|| 2);

    // Cases when unwrap is not called on a simple variable
    let _ = Some(10).unwrap_or_else(|| 2);
    let _ = Some(10).and_then(|_| ext_opt);
    let _: Option<u32> = None.or_else(|| ext_opt);
    let _ = None.get_or_insert_with(|| 2);
    let _: Result<u32, u32> = None.ok_or_else(|| 2);
    let _: Option<u32> = None.or_else(|| None);

    let mut deep = Deep(Some(42));
    let _ = deep.0.unwrap_or_else(|| 2);
    let _ = deep.0.and_then(|_| ext_opt);
    let _ = deep.0.or_else(|| None);
    let _ = deep.0.get_or_insert_with(|| 2);
    let _ = deep.0.ok_or_else(|| 2);

    // Should not lint - Option
    let _ = opt.unwrap_or_else(|| ext_str.return_some_field());
    let _ = opt.or_else(some_call);
    let _ = opt.or_else(|| some_call());
    let _: Result<u32, u32> = opt.ok_or_else(|| some_call());
    let _: Result<u32, u32> = opt.ok_or_else(some_call);
    let _ = deep.0.get_or_insert_with(|| some_call());
    let _ = deep.0.or_else(some_call);
    let _ = deep.0.or_else(|| some_call());

    // These are handled by bind_instead_of_map
    let _: Option<u32> = None.or_else(|| Some(3));
    let _ = deep.0.or_else(|| Some(3));
    let _ = opt.or_else(|| Some(3));

    // Should lint - Result
    let res: Result<u32, u32> = Err(5);
    let res2: Result<u32, SomeStruct> = Err(SomeStruct { some_field: 5 });

    let _ = res2.unwrap_or_else(|_| 2);
    let _ = res2.unwrap_or_else(|_| astronomers_pi);
    let _ = res2.unwrap_or_else(|_| ext_str.some_field);

    // Should not lint - Result
    let _ = res.unwrap_or_else(|err| err);
    let _ = res2.unwrap_or_else(|err| err.some_field);
    let _ = res2.unwrap_or_else(|err| err.return_some_field());
    let _ = res2.unwrap_or_else(|_| ext_str.return_some_field());

    let _: Result<u32, u32> = res.and_then(|x| Ok(x));
    let _: Result<u32, u32> = res.and_then(|x| Err(x));

    let _: Result<u32, u32> = res.or_else(|err| Ok(err));
    let _: Result<u32, u32> = res.or_else(|err| Err(err));

    // These are handled by bind_instead_of_map
    let _: Result<u32, u32> = res.and_then(|_| Ok(2));
    let _: Result<u32, u32> = res.and_then(|_| Ok(astronomers_pi));
    let _: Result<u32, u32> = res.and_then(|_| Ok(ext_str.some_field));

    let _: Result<u32, u32> = res.and_then(|_| Err(2));
    let _: Result<u32, u32> = res.and_then(|_| Err(astronomers_pi));
    let _: Result<u32, u32> = res.and_then(|_| Err(ext_str.some_field));

    let _: Result<u32, u32> = res.or_else(|_| Ok(2));
    let _: Result<u32, u32> = res.or_else(|_| Ok(astronomers_pi));
    let _: Result<u32, u32> = res.or_else(|_| Ok(ext_str.some_field));

    let _: Result<u32, u32> = res.or_else(|_| Err(2));
    let _: Result<u32, u32> = res.or_else(|_| Err(astronomers_pi));
    let _: Result<u32, u32> = res.or_else(|_| Err(ext_str.some_field));
}
