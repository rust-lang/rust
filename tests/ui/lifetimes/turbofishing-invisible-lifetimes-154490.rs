trait EvilTrait {
    type EvilAssoc<'a>;
}

// zero explicit generic lifetimes
fn evil_early_bound_1<T: EvilTrait>(_: Option<<T as EvilTrait>::EvilAssoc<'_>>) -> &i32 {
    todo!()
}

fn evil_early_bound_2<T: EvilTrait>(_: Option<<T as EvilTrait>::EvilAssoc<'_>>) -> &i32 {
    todo!()
}

fn normal_early_bound<'eb: 'eb, T: EvilTrait>(_: &'eb i32 ) -> &'eb i32 {
    todo!()
}

fn normal_late_bound<'lb, T: EvilTrait>(_: &'lb i32 ) -> &'lb i32 {
    todo!()
}

struct LtWrapper<'a>(&'a i32);

fn elide_struct_1(_: &i32) -> LtWrapper {
    todo!()
}

fn elide_struct_2(_: &i32) -> LtWrapper {
    todo!()
}

fn foo<T: EvilTrait>() {
    static WHATEVER: i32 = 123;

    evil_early_bound_1::<'static, T>(None);
    //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    // ^ FIXME: should this have a better diagnostic?

    elide_struct_1(&WHATEVER);

    elide_struct_2::<'static>(&WHATEVER);
    //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

    evil_early_bound_2::<T>(None);

    normal_early_bound::<'static, T>(&WHATEVER);
    // ^ this is fine

    normal_late_bound::<'static, T>(&WHATEVER);
    //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
}


fn main() {}
